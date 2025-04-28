# Version 3.0.0: Defines the trainable head for feature sequences.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Added math
import functools # Needed for RMSNorm partial


# --- Use RMSNorm or LayerNorm ---
# Let's default back to LayerNorm for simplicity unless specified otherwise
RMSNorm = nn.RMSNorm
Use_RMSNorm = True # Set to True to use RMSNorm/SwiGLU experiment
NormLayer = nn.RMSNorm if Use_RMSNorm else nn.LayerNorm
print(f"DEBUG HeadModel: Using {'RMSNorm' if Use_RMSNorm else 'LayerNorm'}")


# --- SwiGLUFFN adapted for HeadModel MLP ---
# This replaces a standard Linear -> Activation -> Linear block
class SwiGLUFFNHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, act_layer: nn.Module = nn.SiLU, dropout: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Project to hidden dim * 2 for SwiGLU gate and value
        self.w12 = nn.Linear(in_features, hidden_features * 2, bias=False)
        self.act = act_layer()
        self.dropout1 = nn.Dropout(dropout) # Dropout after activation * value
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)
        self.dropout2 = nn.Dropout(dropout) # Dropout before final output

    def forward(self, x):
        # Split the output of the first linear layer into two parts
        gate, value = self.w12(x).chunk(2, dim=-1)
        # Apply activation to gate, multiply by value, apply dropout
        x = self.dropout1(self.act(gate) * value)
        # Apply final linear layer and dropout
        x = self.dropout2(self.w3(x))
        return x


# --- ResBlock (Conditional Norm/Activation) ---
# v1.7.0: Conditional Norm/Activation
class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.norm = NormLayer(ch) # Use chosen NormLayer
        if Use_RMSNorm:
            # Use SwiGLUFFNHead
            self.ffn = SwiGLUFFNHead(in_features=ch, dropout=dropout)
        else:
            # Use original Linear + GELU structure
            self.ffn = nn.Sequential(
                nn.Linear(ch, ch * 4), # Maybe wider FFN here? Original was ch*1
                nn.GELU(),
                nn.Dropout(dropout), # Add dropout within FFN?
                nn.Linear(ch * 4, ch),
                 nn.Dropout(dropout) # Dropout after FFN?
            )
            # Original ResBlock was: Linear(ch,ch), GELU, Linear(ch,ch), GELU, Linear(ch,ch) - simpler FFN. Let's use that.
            # self.ffn = nn.Sequential(
            #     nn.Linear(ch, ch), nn.GELU(),
            #     nn.Linear(ch, ch), nn.GELU(),
            #     nn.Linear(ch, ch),
            #     # No dropout in original ResBlock FFN
            # )

    def forward(self, x):
        # return x + self.gamma * self.ffn(self.norm(x)) # With LayerScale
        return x + self.ffn(self.norm(x)) # Without LayerScale


# --- Attention Pooling Layer (v1.0.0 - Unchanged) ---
class AttentionPool(nn.Module):
    """Pools a sequence of features using self-attention."""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
    # v1.1.0: Accept and use key_padding_mask
    def forward(self, x, key_padding_mask=None): # Add mask argument
        batch_size = x.shape[0]
        query = self.query_token.expand(batch_size, -1, -1)
        # <<< Pass key_padding_mask to attention layer >>>
        # Mask should be True where padded (ignored), False where valid. Our mask is opposite.
        # So we need to invert it IF MHA expects True for padding. Check docs.
        # PyTorch MHA: key_padding_mask (bool): If specified, a mask indicating True for positions to be masked (padded).
        # Our mask: True for real tokens, False for padding. So, invert it:
        inverted_mask = ~key_padding_mask if key_padding_mask is not None else None

        attn_output, _ = self.attention(
            query=query, key=x, value=x,
            key_padding_mask=inverted_mask # Pass the inverted mask
        )
        # <<< End Mask Passing >>>
        pooled_output = attn_output.squeeze(1)
        pooled_output = self.norm(pooled_output)
        return pooled_output


# --- Head Model Definition ---
# --- Head Model Definition (No Transformer Layers) ---
# v1.7.0: Removed TransformerEncoder, uses pooling strategy directly on input
class HeadModel(nn.Module):
    def __init__(self,
                 features: int,
                 num_classes: int,
                 # --- Pooling Strategy is key now ---
                 pooling_strategy: str = 'attn', # 'attn', 'avg', 'none' (for pre-pooled)
                 # --- MLP Params ---
                 hidden_dim: int = 1024,
                 num_res_blocks: int = 3,
                 dropout_rate: float = 0.2,
                 output_mode: str = 'linear',
                 # --- Attn Pool Params (used if pooling_strategy='attn') ---
                 attn_pool_heads: int = 16,
                 attn_pool_dropout: float = 0.2
                 ):
        super().__init__()
        self.pooling_strategy = pooling_strategy.lower()
        self.output_mode = output_mode.lower()
        self.num_classes = num_classes
        current_features = features

        print(f"Initializing HeadModel v1.7.0 (Pool + MLP):")
        print(f"  Input Features: {features}, Pooling: {self.pooling_strategy}")
        print(f"  MLP Hidden: {hidden_dim}, ResBlocks: {num_res_blocks}, Dropout: {dropout_rate}, Norm: {'RMSNorm' if Use_RMSNorm else 'LayerNorm'}")
        print(f"  Output Mode: {self.output_mode}, Classes: {self.num_classes}")

        # --- Optional Pooling Layer (Applied to input sequence) ---
        self.pooler = None
        if self.pooling_strategy == 'attn':
             # --- Optional: Add head adjustment logic if needed ---
             # This ensures 'num_heads' is valid for the 'features' dimension
             actual_attn_heads = attn_pool_heads # Start with configured value
             if features % attn_pool_heads != 0:
                  possible_heads = [h for h in [1, 2, 4, 8, 16, 32] if features % h == 0] # Common head counts
                  if not possible_heads: raise ValueError(f"Features ({features}) not divisible by any standard head count.")
                  actual_attn_heads = min(possible_heads, key=lambda x:abs(x-attn_pool_heads))
                  print(f"  Warning: Adjusting pooling attn heads from {attn_pool_heads} to {actual_attn_heads} for features={features}.")
             # --- End head adjustment logic ---

             # Create the AttentionPool instance
             self.pooler = AttentionPool(
                 embed_dim=features,
                 num_heads=actual_attn_heads, # Use potentially adjusted head count
                 dropout=attn_pool_dropout  # Use dropout from attn_pool_params
             )
             print(f"  Using Attention Pooling on input sequence (Heads: {actual_attn_heads}).")
        elif self.pooling_strategy not in ['avg', 'none']:
             print(f"Warning: Unknown pooling strategy '{self.pooling_strategy}'. Defaulting to average pooling.")
             self.pooling_strategy = 'avg'

        # --- MLP Head (Applied AFTER pooling) ---
        mlp_layers = []
        mlp_layers.append(NormLayer(features)) # <<< Norm AFTER pooling >>>
        mlp_layers.append(nn.Linear(features, hidden_dim)) # Initial projection
        if not Use_RMSNorm: mlp_layers.append(nn.GELU()) # Add GELU if not using SwiGLU in ResBlocks
        mlp_layers.append(nn.Dropout(dropout_rate)) # Dropout after initial projection? Or inside ResBlock?

        # Use ResBlocks (updated version using NormLayer and maybe SwiGLU)
        for _ in range(num_res_blocks):
            mlp_layers.append(ResBlock(ch=hidden_dim, dropout=dropout_rate)) # Pass dropout

        # Down projection (Conditional Norm/Activation)
        if Use_RMSNorm:
             mlp_layers.extend([
                  RMSNorm(hidden_dim),
                  SwiGLUFFNHead(hidden_dim, hidden_features=hidden_dim//2, out_features=hidden_dim // 2, dropout=dropout_rate/2),
                  RMSNorm(hidden_dim // 2),
                  nn.Linear(hidden_dim // 2, self.num_classes)
             ])
        else: # Use LayerNorm / GELU
             mlp_layers.extend([
                 NormLayer(hidden_dim), # Norm before down proj
                 nn.Linear(hidden_dim, hidden_dim // 2),
                 nn.GELU(),
                 nn.Dropout(dropout_rate / 2),
                 NormLayer(hidden_dim // 2), # Norm after down proj? Or before final linear? Before.
                 nn.Linear(hidden_dim // 2, self.num_classes)
             ])

        self.mlp_head = nn.Sequential(*mlp_layers)

        # --- Validate Output Mode ---
        valid_modes = ['linear', 'sigmoid', 'softmax', 'tanh_scaled']
        if self.output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{self.output_mode}'. Must be one of {valid_modes}")
        if self.output_mode == 'softmax' and self.num_classes <= 1:
             print(f"  Warning: output_mode='softmax' usually used with num_classes > 1 (got {self.num_classes}).")
        if self.output_mode in ['sigmoid', 'tanh_scaled'] and self.num_classes != 1:
             print(f"  Warning: output_mode='{self.output_mode}' usually used with num_classes=1 (got {self.num_classes}).")

    def forward(self, sequence, attention_mask=None):
        # Input sequence shape: [B, SeqLen, Features]
        # Input attention_mask shape: [B, SeqLen] (True=Real, False=Pad)

        # 1. Apply Pooling Strategy
        features = None # Initialize
        if self.pooling_strategy == 'attn':
             if self.pooler is None: raise ValueError("Attn pooling selected but pooler not initialized.")
             features = self.pooler(sequence, key_padding_mask=attention_mask) # [B, F]
        elif self.pooling_strategy == 'avg':
             # Average only non-padded tokens
             if attention_mask is not None:
                  mask_expanded = attention_mask.unsqueeze(-1).expand_as(sequence)
                  masked_sum = torch.sum(sequence * mask_expanded.float(), dim=1)
                  num_real_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                  features = masked_sum / num_real_tokens # [B, F]
             else: # No mask, average all
                  features = torch.mean(sequence, dim=1) # [B, F]
        elif self.pooling_strategy == 'none':
             # Assumes input 'sequence' is already pooled [B, F]
             features = sequence
             if features.ndim != 2: raise ValueError(f"Pooling strategy is 'none' but input sequence has {features.ndim} dims (expected 2).")
        else:
             raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Optional: Normalize pooled features? (Remove if features are normalized during generation)
        # features = F.normalize(features, p=2, dim=-1)

        # 2. Pass pooled features through MLP Head
        logits = self.mlp_head(features)

        # --- Apply Activation (Unchanged) ---
        output = None
        if self.output_mode == 'linear': output = logits
        elif self.output_mode == 'sigmoid': output = torch.sigmoid(logits)
        elif self.output_mode == 'softmax': output = F.softmax(logits, dim=-1)
        elif self.output_mode == 'tanh_scaled': output = (torch.tanh(logits) + 1.0) / 2.0
        else: raise RuntimeError(f"Invalid output_mode '{self.output_mode}'.")

        if self.num_classes == 1 and output.ndim == 2 and output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output