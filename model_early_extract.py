# model_early_extract.py
# Version 1.0.0: End-to-end model with early feature extraction & attention pooling

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig
import math

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
    def __init__(self,
                 base_model_name: str, # e.g., "google/siglip2-so400m-patch16-naflex"
                 # --- Feature Extraction Params ---
                 extract_layer: int = -1, # Which hidden layer to use (-1 for last)
                 pooling_strategy: str = 'attn', # 'cls', 'avg', 'attn'
                 # --- Predictor Head Params ---
                 # Feature dim often matches base model hidden dim, but pooling might change it
                 head_features: int = None, # If None, infer from base model
                 head_hidden_dim: int = 1024,
                 head_num_classes: int = 2,
                 head_num_res_blocks: int = 2,
                 head_dropout_rate: float = 0.2,
                 head_output_mode: str = 'linear',
                 # --- Attention Pooling Params (if pooling_strategy='attn') ---
                 attn_pool_heads: int = 8,
                 attn_pool_dropout: float = 0.1,
                 # --- Fine-tuning Params ---
                 freeze_base_model: bool = True,
                 # --- Other ---
                 compute_dtype: torch.dtype = torch.float32
                 ):
        super().__init__()

        self.extract_layer = extract_layer
        self.pooling_strategy = pooling_strategy.lower()
        self.compute_dtype = compute_dtype

        # --- 1. Load Base Vision Model ---
        print(f"Loading base vision model: {base_model_name}")
        try:
            # Load config first to get hidden size etc.
            vision_config = PretrainedConfig.from_pretrained(base_model_name, trust_remote_code=True)
            # Try finding hidden size (might be under vision_config or directly)
            self.base_hidden_dim = getattr(vision_config, 'hidden_size', getattr(getattr(vision_config, 'vision_config', {}), 'hidden_size', None))
            if self.base_hidden_dim is None:
                 # Add specific checks for models like DINOv2 if needed
                 if hasattr(vision_config, 'hidden_dim'): self.base_hidden_dim = vision_config.hidden_dim
                 else: raise ValueError("Could not determine base model hidden dimension.")

            self.vision_model = AutoModel.from_pretrained(
                base_model_name,
                torch_dtype=self.compute_dtype,
                trust_remote_code=True,
                # Add output_hidden_states=True if needed by pooling strategy
                output_hidden_states=(self.pooling_strategy != 'pooler') # Pooler output doesn't need hidden states
            )
            print(f"  Base model loaded. Hidden Dim: {self.base_hidden_dim}")
        except Exception as e:
            raise RuntimeError(f"Failed to load base vision model {base_model_name}: {e}")

        # --- Freeze Base Model (Optional) ---
        if freeze_base_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval() # Set base to eval if frozen
            print("  Base vision model frozen.")
        else:
            self.vision_model.train() # Ensure base is trainable if not frozen

        # --- Determine Input Features for Head ---
        if head_features is None:
            # If using CLS token or Avg/Attn pooling on hidden states, feature dim = base hidden dim
            if self.pooling_strategy in ['cls', 'avg', 'attn']:
                self.head_features = self.base_hidden_dim
            elif self.pooling_strategy == 'pooler':
                # Pooler output size might differ (check config again)
                pooler_dim = getattr(vision_config, 'projection_dim', self.base_hidden_dim) # Example check
                self.head_features = pooler_dim
            else:
                 raise ValueError(f"Cannot determine head features for pooling strategy '{self.pooling_strategy}'. Please specify head_features.")
            print(f"  Inferred head input features: {self.head_features}")
        else:
             self.head_features = head_features # Use provided value
             print(f"  Using specified head input features: {self.head_features}")

        # --- 2. Define Pooling Layer (if needed) ---
        self.pooler = None
        if self.pooling_strategy == 'attn':
             if self.head_features != self.base_hidden_dim: print("Warning: Attention pooling input dim mismatch? Using base_hidden_dim.")
             self.pooler = AttentionPool(
                 embed_dim=self.base_hidden_dim, # Attn operates on base hidden dim
                 num_heads=attn_pool_heads,
                 dropout=attn_pool_dropout
             )
             print("  Using Attention Pooling layer.")
        elif self.pooling_strategy not in ['cls', 'avg', 'pooler']:
             raise ValueError(f"Invalid pooling_strategy: '{self.pooling_strategy}'. Choose 'cls', 'avg', 'attn', or 'pooler'.")

        # --- 3. Define Predictor Head ---
        # Slightly simplified head structure example (can reuse PredictorModel if adapted)
        print("Initializing Predictor Head...")
        self.head = nn.Sequential(
            nn.LayerNorm(self.head_features), # Normalize input to head
            nn.Linear(self.head_features, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout_rate),
            # Optional ResBlocks (could add loop here)
            *[ResBlock(ch=head_hidden_dim) for _ in range(head_num_res_blocks)],
            # Final layers
            nn.Linear(head_hidden_dim, head_hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(head_dropout_rate),
            nn.Linear(head_hidden_dim // 4, head_num_classes)
        )
        self.head_output_mode = head_output_mode.lower()
        print(f"  Head initialized. Input: {self.head_features}, Output: {head_num_classes}, Mode: {self.head_output_mode}")

        # --- 4. Optional: Final L2 Norm (if needed for specific base models) ---
        self.needs_final_norm = "dinov2" in base_model_name.lower() and self.pooling_strategy != 'pooler'
        if self.needs_final_norm: print("  Will apply final L2 norm (DINOv2 detected).")


    def forward(self, pixel_values, attention_mask=None, spatial_shapes=None): # Match potential inputs

        # --- Run Base Vision Model ---
        # Prepare inputs based on model type (DINOv2 doesn't use mask/shapes usually)
        model_kwargs = {"pixel_values": pixel_values.to(dtype=self.compute_dtype)}
        if attention_mask is not None and not self.needs_final_norm: # Pass mask if not DINO
             model_kwargs["attention_mask"] = attention_mask
        if spatial_shapes is not None and not self.needs_final_norm: # Pass shapes if not DINO
             model_kwargs["spatial_shapes"] = spatial_shapes # Assuming tensor already

        outputs = self.vision_model(**model_kwargs)

        # --- Extract Features based on Strategy ---
        features = None
        if self.pooling_strategy == 'cls':
            # Assumes last_hidden_state is available and CLS is first token
            last_hidden = outputs.last_hidden_state # Shape: [B, SeqLen, Hidden]
            features = last_hidden[:, 0] # Take CLS token -> [B, Hidden]
        elif self.pooling_strategy == 'avg':
            # Assumes last_hidden_state is available
            last_hidden = outputs.last_hidden_state # Shape: [B, SeqLen, Hidden]
            # Average pool patch tokens (optional: exclude CLS token?)
            features = torch.mean(last_hidden[:, 1:], dim=1) # Avg pool sequence dim -> [B, Hidden]
        elif self.pooling_strategy == 'attn':
            # Assumes last_hidden_state is available
            last_hidden = outputs.last_hidden_state # Shape: [B, SeqLen, Hidden]
            features = self.pooler(last_hidden) # Apply attention pooling -> [B, HeadFeatures]
        elif self.pooling_strategy == 'pooler':
            # Use the default pooled output if available
            if not hasattr(outputs, 'pooler_output') or outputs.pooler_output is None:
                 # Fallback needed if model doesn't have pooler (like some DINOv2)
                 print("Warning: Pooling strategy 'pooler' selected but no pooler_output found. Falling back to CLS token.")
                 last_hidden = outputs.last_hidden_state
                 features = last_hidden[:, 0]
            else:
                 features = outputs.pooler_output # Shape: [B, PoolerDim]
        else:
             # Should be caught in init
             raise RuntimeError("Invalid pooling strategy in forward pass.")

        # --- Optional Final L2 Normalization ---
        if self.needs_final_norm:
             features = F.normalize(features.float(), p=2, dim=-1).to(dtype=self.compute_dtype)

        # --- Run Predictor Head ---
        # Head expects FP32 input? Or match compute_dtype? Let's try FP32 for stability.
        logits = self.head(features.to(torch.float32))

        # --- Apply Head Output Activation ---
        if self.head_output_mode == 'linear':
            output = logits
        elif self.head_output_mode == 'sigmoid':
            output = torch.sigmoid(logits)
        elif self.head_output_mode == 'softmax':
            output = F.softmax(logits, dim=-1)
        elif self.head_output_mode == 'tanh_scaled':
            output = (torch.tanh(logits) + 1.0) / 2.0
        else:
            raise RuntimeError(f"Invalid head_output_mode '{self.head_output_mode}'.")

        # Squeeze if single class output
        if output.shape[-1] == 1 and output.ndim > 1:
            output = output.squeeze(-1)

        return output