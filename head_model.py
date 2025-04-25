# Version 1.0.0: Defines the trainable head for feature sequences.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Added math

# --- ResBlock Definition (v1.0.0 - Unchanged for now) ---
class ResBlock(nn.Module):
    """Standard Residual Block with LayerNorm."""
    def __init__(self, ch): # No scale_factor for now
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.long = nn.Sequential(
            nn.Linear(ch, ch), nn.GELU(),
            nn.Linear(ch, ch), nn.GELU(),
            nn.Linear(ch, ch),
        )
    def forward(self, x):
        return x + self.long(self.norm(x))

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
    def forward(self, x):
        batch_size = x.shape[0]
        query = self.query_token.expand(batch_size, -1, -1)
        attn_output, _ = self.attention(query=query, key=x, value=x)
        pooled_output = attn_output.squeeze(1)
        pooled_output = self.norm(pooled_output)
        return pooled_output

# --- Head Model Definition ---
# v1.1.0: Added L2 norm after pooling, less aggressive downprojection
class HeadModel(nn.Module):
    """
    Trainable Head (Pooler + MLP) for processing pre-computed feature sequences.
    Takes sequence input [Batch, NumPatches, Features] and outputs predictions.
    v1.1.0: Added L2 norm after pooling, less aggressive downprojection.
    """
    def __init__(self,
                 features: int,
                 num_classes: int,
                 pooling_strategy: str = 'attn',
                 hidden_dim: int = 1024,
                 num_res_blocks: int = 3,
                 dropout_rate: float = 0.2,
                 output_mode: str = 'linear',
                 attn_pool_heads: int = 16,
                 attn_pool_dropout: float = 0.2
                 ):
        super().__init__()
        self.pooling_strategy = pooling_strategy.lower()
        self.output_mode = output_mode.lower()
        self.num_classes = num_classes
        current_features = features

        # --- Logging (Unchanged) ---
        print(f"Initializing HeadModel v1.1.0:") # Version bump
        print(f"  Input Features: {features}, Pooling: {self.pooling_strategy}")
        print(f"  MLP Hidden: {hidden_dim}, ResBlocks: {num_res_blocks}, Dropout: {dropout_rate}")
        print(f"  Output Mode: {self.output_mode}, Classes: {self.num_classes}")
        print(f"  Changes: Added L2 Norm after pooling, Downprojection uses hidden_dim//2") # Log changes

        # --- Optional Pooling Layer (Unchanged) ---
        self.pooler = None
        if self.pooling_strategy == 'attn':
             actual_attn_heads = attn_pool_heads
             if features % attn_pool_heads != 0:
                  possible_heads = [h for h in [1, 2, 4, 8, 16, 32] if features % h == 0]
                  if not possible_heads: raise ValueError(f"Features ({features}) not divisible by any standard head count.")
                  actual_attn_heads = min(possible_heads, key=lambda x:abs(x-attn_pool_heads))
                  print(f"  Warning: Adjusting pooling attn heads from {attn_pool_heads} to {actual_attn_heads} for features={features}.")
             self.pooler = AttentionPool(embed_dim=features, num_heads=actual_attn_heads, dropout=attn_pool_dropout)
             print(f"  Using Attention Pooling (Heads: {actual_attn_heads}, Dropout: {attn_pool_dropout})")
        elif self.pooling_strategy not in ['avg', 'none']:
             print(f"Warning: Unknown pooling strategy '{self.pooling_strategy}'. Defaulting to average pooling.")
             self.pooling_strategy = 'avg'

        # --- MLP Head ---
        print("  Initializing MLP Head (v1.1.0)...")
        self.head = nn.Sequential(
            nn.LayerNorm(current_features), # Norm applied *before* projection now
            nn.Linear(current_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate), # Dropout after initial projection
            *[ResBlock(ch=hidden_dim) for _ in range(num_res_blocks)],
            # --- MODIFIED Down projection part ---
            nn.Linear(hidden_dim, hidden_dim // 2), # <<< CHANGED: Less aggressive reduction
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2), # <<< Optional: Reduce dropout here slightly? Default 0.1 if rate is 0.2
            nn.Linear(hidden_dim // 2, self.num_classes) # <<< CHANGED: Final output layer
            # --- END MODIFICATION ---
        )

        # --- Validate Output Mode (Unchanged) ---
        valid_modes = ['linear', 'sigmoid', 'softmax', 'tanh_scaled']
        if self.output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{self.output_mode}'. Must be one of {valid_modes}")
        # Warnings about num_classes vs mode remain the same...

    def forward(self, sequence):
        # Input sequence shape: [BatchSize, NumPatches, Features]
        features = sequence

        # --- Apply Pooling ---
        if features.ndim == 3: # Only pool if it's a sequence
            if self.pooler is not None: features = self.pooler(features) # Output: [BatchSize, Features]
            elif self.pooling_strategy == 'avg': features = torch.mean(features, dim=1) # Average over patches -> [BatchSize, Features]
            # If strategy is 'none', features should already be [BatchSize, Features]
        elif features.ndim != 2:
             raise ValueError(f"Unexpected input dimension to HeadModel forward: {features.ndim}. Expected 2 or 3.")

        # <<< ADDED: Apply L2 Normalization AFTER Pooling >>>
        # Normalize the pooled features before passing to the head MLP
        features = F.normalize(features, p=2, dim=-1) # Normalize along the feature dimension

        # --- Apply Head ---
        logits = self.head(features) # Input `features` is now normalized [BatchSize, Features]

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