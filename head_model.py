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
# v1.5.0: Use Transformer Encoder layers before pooling
class HeadModel(nn.Module):
    """
    HeadModel using Transformer Encoder layers before pooling.
    """
    def __init__(self,
                 features: int,
                 num_classes: int,
                 # --- NEW Params ---
                 num_transformer_layers: int = 1, # How many transformer layers to add
                 transformer_nhead: int = 8,      # Heads for the new transformer layers
                 transformer_dim_feedforward: int = None, # Dim for transformer FFN (default: features*2 or 4)
                 transformer_dropout: float = 0.1,
                 # --- Pooling after Transformer ---
                 pooling_after_transformer: str = 'avg', # 'avg', 'cls', 'attn'
                 # --- Existing MLP Params ---
                 hidden_dim: int = 1024,        # Hidden dim for MLP *after* pooling
                 num_res_blocks: int = 3,       # ResBlocks in MLP
                 dropout_rate: float = 0.2,     # Dropout in MLP
                 output_mode: str = 'linear',
                 # --- Old Pooling Params (only used if pooling_after_transformer='attn') ---
                 attn_pool_heads: int = 16,
                 attn_pool_dropout: float = 0.2
                 ):
        super().__init__()
        self.output_mode = output_mode.lower()
        self.num_classes = num_classes
        self.pooling_after_transformer = pooling_after_transformer.lower()

        print(f"Initializing HeadModel v1.5.0 (Transformer Head):")
        print(f"  Input Features: {features}")
        print(f"  Transformer Layers: {num_transformer_layers}, Heads: {transformer_nhead}, Dropout: {transformer_dropout}")
        print(f"  Pooling After Transformer: {self.pooling_after_transformer}")
        print(f"  MLP Hidden: {hidden_dim}, ResBlocks: {num_res_blocks}, Dropout: {dropout_rate}")
        print(f"  Output Mode: {self.output_mode}, Classes: {self.num_classes}")

        # --- Transformer Encoder Layers ---
        if transformer_dim_feedforward is None:
            transformer_dim_feedforward = features * 2 # A smaller FFN dim might be faster

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='gelu', # Or 'relu'
            batch_first=True,
            norm_first=True # Pre-LN is often more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- Pooling Layer (Optional: only if pooling_after_transformer='attn') ---
        self.pooler = None
        if self.pooling_after_transformer == 'attn':
             self.pooler = AttentionPool(
                 embed_dim=features,
                 num_heads=attn_pool_heads, # Use separate head count for pooling?
                 dropout=attn_pool_dropout
             )
             print(f"  Using Attention Pooling after Transformer")

        # --- MLP Head (Takes features dimension as input) ---
        # Input dim to MLP is always `features` after pooling
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(features), # Norm the pooled features
            nn.Linear(features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            *[ResBlock(ch=hidden_dim) for _ in range(num_res_blocks)],
            # Use the less aggressive bottleneck
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, self.num_classes)
        )

        # --- Validate Output Mode (Unchanged) ---
        valid_modes = ['linear', 'sigmoid', 'softmax', 'tanh_scaled']
        if self.output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{self.output_mode}'. Must be one of {valid_modes}")
        # Warnings about num_classes vs mode remain the same...

    def forward(self, sequence, attention_mask=None):
        # Input sequence shape: [B, SeqLen, Features] (e.g., [B, 4096, 1024])
        # Input attention_mask shape: [B, SeqLen] (Bool: True=Real, False=Pad)

        # Invert mask for TransformerEncoderLayer (True = Ignore)
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None

        # 1. Pass through Transformer Encoder Layers
        # Input needs to be Float32 potentially? Or handle via AMP. Assume AMP handles.
        transformer_output = self.transformer_encoder(sequence, src_key_padding_mask=src_key_padding_mask)
        # Output shape: [B, SeqLen, Features]

        # 2. Pool the output sequence
        pooled_features = None
        if self.pooling_after_transformer == 'avg':
             # Need to handle mask for averaging correctly! Average only non-padded tokens.
             if attention_mask is not None:
                  mask_expanded = attention_mask.unsqueeze(-1).expand_as(transformer_output) # [B, SeqLen, F]
                  masked_sum = torch.sum(transformer_output * mask_expanded.float(), dim=1) # Sum only real tokens
                  num_real_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1) # Count real tokens [B, 1]
                  pooled_features = masked_sum / num_real_tokens # [B, F]
             else: # No mask provided
                  pooled_features = torch.mean(transformer_output, dim=1) # Simple average [B, F]

        elif self.pooling_after_transformer == 'cls':
             pooled_features = transformer_output[:, 0, :] # Take the output corresponding to the first token [B, F]
        elif self.pooling_after_transformer == 'attn':
             if self.pooler is None: raise ValueError("Attn pooling selected but pooler not initialized.")
             # AttentionPool expects mask where True=Ignore (inverted mask)
             pooled_features = self.pooler(transformer_output, key_padding_mask=src_key_padding_mask) # [B, F]
        else:
             raise ValueError(f"Unknown pooling_after_transformer strategy: {self.pooling_after_transformer}")

        # 3. Pass pooled features through MLP Head
        logits = self.mlp_head(pooled_features)


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