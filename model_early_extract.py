# model_early_extract.py
# Version 1.1.0: aimv2 is not in transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
# Remove transformers AutoModel import if no longer needed
# from transformers import AutoModel, PretrainedConfig
import math

# <<< Add AIMv2 import >>>
try:
    from aim.v2.utils import load_pretrained
    AIMV2_AVAILABLE = True
except ImportError:
    print("Warning: Could not import 'load_pretrained' from 'aim.v2.utils'.")
    print("Ensure aim.v2 is installed: pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v2'")
    AIMV2_AVAILABLE = False

# --- ResBlock (Can copy from original model.py or redefine) ---
class ResBlock(nn.Module):
    """Standard Residual Block with LayerNorm."""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.long = nn.Sequential(
            nn.Linear(ch, ch), nn.GELU(),
            nn.Linear(ch, ch), nn.GELU(),
            nn.Linear(ch, ch),
        )
    def forward(self, x):
        return x + self.long(self.norm(x))

# --- Attention Pooling Layer ---
class AttentionPool(nn.Module):
    """
    Pools a sequence of features using self-attention.
    Creates a learnable query token, attends to the input sequence, returns pooled output.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable query token (similar to CLS token idea but learns pooling)
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Maybe add a small MLP after attention? Optional.
        # self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, x):
        # x shape: [Batch, SequenceLength, EmbedDim]
        batch_size = x.shape[0]
        # Expand query token to batch size: [Batch, 1, EmbedDim]
        query = self.query_token.expand(batch_size, -1, -1)

        # Attention: Query attends to the input sequence (x)
        # Q = query, K = x, V = x
        attn_output, _ = self.attention(query, x, x)
        # attn_output shape: [Batch, 1, EmbedDim]

        # Pool by taking the output corresponding to the query token
        pooled_output = attn_output.squeeze(1) # Shape: [Batch, EmbedDim]

        # Apply LayerNorm (and optional MLP)
        pooled_output = self.norm(pooled_output)
        # if hasattr(self, 'mlp'): pooled_output = self.mlp(pooled_output)

        return pooled_output

# --- Main End-to-End Model ---
class EarlyExtractAnatomyModel(nn.Module):
    # <<< Add device parameter >>>
    def __init__(self,
                 base_model_name: str,
                 device: str = "cpu", # <<< Added >>>
                 # --- Feature Extraction Params ---
                 extract_layer: int = -1,
                 pooling_strategy: str = 'attn',
                 # --- Predictor Head Params ---
                 head_features: int = None,
                 head_hidden_dim: int = 1024,
                 head_num_classes: int = 2,
                 head_num_res_blocks: int = 2,
                 head_dropout_rate: float = 0.2,
                 head_output_mode: str = 'linear',
                 # --- Attention Pooling Params ---
                 attn_pool_heads: int = 8,
                 attn_pool_dropout: float = 0.1,
                 # --- Fine-tuning Params ---
                 freeze_base_model: bool = True,
                 # --- Other ---
                 compute_dtype: torch.dtype = torch.bfloat16
                 ):
        super().__init__()

        self.extract_layer = extract_layer
        self.pooling_strategy = pooling_strategy.lower()
        self.compute_dtype = compute_dtype
        self.device = device # Store device


        # --- 1. Load Base Vision Model ---
        aimv2_short_name = base_model_name.split('/')[-1] # Get part after "apple/"
        print(f"Loading base vision model using aim.v2: {base_model_name} (using short name: {aimv2_short_name})")
        if not AIMV2_AVAILABLE:
            exit("Error: aim.v2 package not available. Cannot load AIMv2 model.")

        try:
            # Use the official AIMv2 loading function
            self.vision_model = load_pretrained(
                aimv2_short_name,
                backend="torch",
            )
            # Move to device and set dtype AFTER loading
            self.vision_model = self.vision_model.to(device=self.device, dtype=self.compute_dtype)

            # Manually configure output_hidden_states if needed
            if hasattr(self.vision_model, 'config'):
                 # Default AIMv2 forward outputs hidden states, so maybe not needed?
                 # Check AIMv2Model forward signature if issues arise.
                 # Let's assume default is okay unless pooling requires specific hidden states.
                 self.vision_model.config.output_hidden_states = True # Ensure it's true if needed
                 print(f"  Set vision_model.config.output_hidden_states = {self.vision_model.config.output_hidden_states}")
            else:
                 print("  Warning: Cannot access vision_model.config to set output_hidden_states.")

            # Get hidden dim
            if hasattr(self.vision_model, 'config') and hasattr(self.vision_model.config, 'hidden_size'):
                self.base_hidden_dim = self.vision_model.config.hidden_size
            else:
                 try: # Attempt fallback using known structure (less robust)
                      self.base_hidden_dim = self.vision_model.trunk.post_trunk_norm.weight.shape[0]
                 except Exception as e_dim:
                      raise ValueError(f"Could not determine base model hidden dimension from loaded AIMv2 model: {e_dim}")

            print(f"  Base model loaded via aim.v2. Hidden Dim: {self.base_hidden_dim}")

        except Exception as e:
            print(f"Details: {e}")
            raise RuntimeError(f"Failed to load base vision model {base_model_name} using aim.v2.")

        # --- Freeze Base Model ---
        if freeze_base_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
            print("  Base vision model frozen.")
        else:
            self.vision_model.train()

        # --- Determine Input Features for Head (logic remains same) ---
        if head_features is None:
            if self.pooling_strategy in ['cls', 'avg', 'attn']: self.head_features = self.base_hidden_dim
            elif self.pooling_strategy == 'pooler':
                 # AIMv2 doesn't really have a 'pooler' output like CLIP/SigLIP
                 print("Warning: Pooling strategy 'pooler' selected for AIMv2. Using 'cls' logic instead (first token).")
                 self.head_features = self.base_hidden_dim
                 self.pooling_strategy = 'cls' # Force change if pooler selected
            else: raise ValueError(f"Cannot determine head features for pooling strategy '{self.pooling_strategy}'.")
            print(f"  Inferred head input features: {self.head_features}")
        else:
             self.head_features = head_features
             print(f"  Using specified head input features: {self.head_features}")


        # --- 2. Define Pooling Layer ---
        self.pooler = None
        if self.pooling_strategy == 'attn':
             if self.head_features != self.base_hidden_dim: print("Warning: Attention pooling input dim mismatch? Using base_hidden_dim.")
             self.pooler = AttentionPool(
                 embed_dim=self.base_hidden_dim, # Attn operates on base hidden dim
                 num_heads=attn_pool_heads, dropout=attn_pool_dropout
             ).to(device=self.device) # Move pooler to device
             print("  Using Attention Pooling layer.")
        elif self.pooling_strategy not in ['cls', 'avg', 'pooler']: # 'pooler' now handled above
             raise ValueError(f"Invalid pooling_strategy: '{self.pooling_strategy}'. Choose 'cls', 'avg', or 'attn'.")


        # --- 3. Define Predictor Head ---
        print("Initializing Predictor Head...")
        self.head = nn.Sequential(
            nn.LayerNorm(self.head_features),
            nn.Linear(self.head_features, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout_rate),
            *[ResBlock(ch=head_hidden_dim) for _ in range(head_num_res_blocks)],
            nn.Linear(head_hidden_dim, head_hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(head_dropout_rate),
            nn.Linear(head_hidden_dim // 4, head_num_classes)
        ).to(device=self.device) # Move head to device
        self.head_output_mode = head_output_mode.lower()
        print(f"  Head initialized. Input: {self.head_features}, Output: {head_num_classes}, Mode: {self.head_output_mode}")

        # --- 4. Optional: Final L2 Norm ---
        self.needs_final_norm = False # AIMv2 features are typically used directly
        # self.needs_final_norm = "dinov2" in base_model_name.lower() and self.pooling_strategy != 'pooler'
        # if self.needs_final_norm: print("  Will apply final L2 norm (DINOv2 detected).")

    # Version 1.1.2: Removed unnecessary detach()
    def forward(self, pixel_values, attention_mask=None, spatial_shapes=None):

        # --- Run Base Vision Model ---
        input_pixels_tensor = pixel_values.to(dtype=self.compute_dtype)
        request_features = (self.pooling_strategy != 'pooler') or (self.extract_layer != -1)
        outputs_tuple = self.vision_model(input_pixels_tensor, output_features=request_features)

        # --- Unpack and Select Base Features ---
        if request_features: # ... (unpack logic) ...
            # ... select base_features_raw from hidden_states_tuple or base_output_raw ...
            if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 2:
                base_output_raw, hidden_states_tuple = outputs_tuple
            else:
                base_output_raw = outputs_tuple; hidden_states_tuple = None

            if self.extract_layer == -1 and hidden_states_tuple: base_features_raw = hidden_states_tuple[-1]
            elif hidden_states_tuple and 0 <= self.extract_layer < len(hidden_states_tuple): base_features_raw = hidden_states_tuple[self.extract_layer]
            else: base_features_raw = base_output_raw # Use output after trunk norm as fallback

        else: # 'pooler' strategy or no features requested
             base_output_raw = outputs_tuple
             hidden_states_tuple = None
             base_features_raw = base_output_raw # Output from vision_model.head is likely already pooled

        if base_features_raw is None: exit("Error: Could not determine base_features_raw.")


        # <<< --- NO DETACH HERE --- >>>
        # Process the features directly, allowing gradients to flow back to pooler if needed
        features_to_pool_or_use = base_features_raw


        # --- Pool Features ---
        pooled_features = None
        # Check if features_to_pool_or_use is already pooled (e.g. from vision_model's internal head)
        # Heuristic: If pooling was 'pooler' or ndim is not 3 (Batch, Seq, Dim)
        is_already_pooled = (self.pooling_strategy == 'pooler' or features_to_pool_or_use.ndim != 3)

        if is_already_pooled:
            pooled_features = features_to_pool_or_use # Use directly
            # print("DEBUG Forward: Using features directly (assumed pre-pooled or not sequence).")
        elif self.pooling_strategy == 'cls':
            # print("Warning: 'cls' pooling not applicable to AIMv2VisionEncoder. Using AVG.")
            pooled_features = torch.mean(features_to_pool_or_use, dim=1)
        elif self.pooling_strategy == 'avg':
            pooled_features = torch.mean(features_to_pool_or_use, dim=1)
            # print("DEBUG Forward: Using AVG pooling.")
        elif self.pooling_strategy == 'attn':
            if self.pooler is not None: pooled_features = self.pooler(features_to_pool_or_use) # <<< Pooler operates on original features >>>
            else: pooled_features = torch.mean(features_to_pool_or_use, dim=1) # Fallback AVG
            # print("DEBUG Forward: Using ATTN pooling.")
        else: # Fallback
            pooled_features = torch.mean(features_to_pool_or_use, dim=1)


        # --- Optional Final L2 Norm ---
        if self.needs_final_norm:
             pooled_features = F.normalize(pooled_features.float(), p=2, dim=-1).to(dtype=self.compute_dtype)

        # --- Run Predictor Head ---
        logits = self.head(pooled_features.to(torch.float32))

        # --- Apply Head Output Activation ---
        if self.head_output_mode == 'linear': output = logits
        elif self.head_output_mode == 'sigmoid': output = torch.sigmoid(logits)
        elif self.head_output_mode == 'softmax': output = F.softmax(logits, dim=-1)
        elif self.head_output_mode == 'tanh_scaled': output = (torch.tanh(logits) + 1.0) / 2.0
        else: raise RuntimeError(f"Invalid head_output_mode '{self.head_output_mode}'.")
        if output.shape[-1] == 1 and output.ndim > 1: output = output.squeeze(-1)

        return output