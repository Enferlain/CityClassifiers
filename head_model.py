# Version 2.0.0: Defines the trainable head for feature sequences.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Added math
import functools # Needed for RMSNorm partial


# --- RMSNorm ---
# Use torch.nn.RMSNorm directly if available (PyTorch 1.11+)
# Otherwise, define it (basic implementation):
# if not hasattr(nn, 'RMSNorm'):
#     class RMSNorm(nn.Module):
#         def __init__(self, dim, eps=1e-6):
#             super().__init__()
#             self.eps = eps
#             self.weight = nn.Parameter(torch.ones(dim))
#         def _norm(self, x):
#             return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         def forward(self, x):
#             output = self._norm(x.float()).type_as(x)
#             return output * self.weight
# else:
# Use the built-in one
RMSNorm = nn.RMSNorm


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


# --- ResBlock (Modified to use RMSNorm, potentially SwiGLU FFN) ---
# v1.6.0: Uses RMSNorm and SwiGLUFFNHead
class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.0): # Added dropout pass-through
        super().__init__()
        self.norm = RMSNorm(ch) # <<< Use RMSNorm >>>
        # Use SwiGLUFFNHead instead of the 3 linear layers + GELU
        # Note: SwiGLUFFNHead includes internal dropout
        self.ffn = SwiGLUFFNHead(in_features=ch, dropout=dropout)
        # Keep residual connection
        # Maybe add LayerScale? (optional, from original ViT/AIM)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

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
# v1.6.0: Uses RMSNorm and SwiGLUFFN in MLP Head
class HeadModel(nn.Module):
    """
    HeadModel using Transformer Encoder layers before pooling.
    """
    def __init__(self, features: int, num_classes: int,
                 num_transformer_layers: int = 1, transformer_nhead: int = 8,
                 transformer_dim_feedforward: int = None, transformer_dropout: float = 0.1,
                 pooling_after_transformer: str = 'avg',
                 hidden_dim: int = 1024, num_res_blocks: int = 3,
                 dropout_rate: float = 0.2, # Main dropout for projections
                 output_mode: str = 'linear',
                 attn_pool_heads: int = 16, attn_pool_dropout: float = 0.2):
        super().__init__()
        self.output_mode = output_mode.lower()
        self.num_classes = num_classes
        self.pooling_after_transformer = pooling_after_transformer.lower()

        print(f"Initializing HeadModel v1.6.0 (Transformer Head + SwiGLU/RMSNorm MLP):")
        print(f"  Input Features: {features}")
        print(
            f"  Transformer Layers: {num_transformer_layers}, Heads: {transformer_nhead}, Dropout: {transformer_dropout}")
        print(f"  Pooling After Transformer: {self.pooling_after_transformer}")
        print(f"  Output Mode: {self.output_mode}, Classes: {self.num_classes}")
        print(f"  MLP using SwiGLU/RMSNorm. Hidden: {hidden_dim}, ResBlocks: {num_res_blocks}, Dropout: {dropout_rate}")

        # --- Transformer Encoder Layers (Keep as is) ---
        if transformer_dim_feedforward is None:
            transformer_dim_feedforward = features * 2  # A smaller FFN dim might be faster

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='gelu',  # Or 'relu'
            batch_first=True,
            norm_first=True  # Pre-LN is often more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- Pooling Layer (Keep as is, optional) ---
        self.pooler = None
        if self.pooling_after_transformer == 'attn': self.pooler = AttentionPool(...) # Same definition

        # --- MLP Head (Modified to use RMSNorm/SwiGLU) ---
        mlp_layers = []
        mlp_layers.append(RMSNorm(features)) # <<< Initial RMSNorm >>>
        mlp_layers.append(nn.Linear(features, hidden_dim)) # Initial projection
        # No activation/dropout here, handled by ResBlocks now

        # Use updated ResBlocks
        for _ in range(num_res_blocks):
            mlp_layers.append(ResBlock(ch=hidden_dim, dropout=dropout_rate)) # Pass dropout to ResBlock's FFN

        # Down projection using SwiGLU FFN blocks? Or keep Linear + Norm + Act?
        # Let's try keeping the Linear -> Norm -> Act structure but use RMSNorm/SiLU
        mlp_layers.extend([
            # nn.Linear(hidden_dim, hidden_dim // 2), # Keep simple projection
            RMSNorm(hidden_dim),                      # <<< RMSNorm before down-projection? Or after? Let's try before like ResBlock.
            SwiGLUFFNHead(hidden_dim, hidden_features=hidden_dim//2, out_features=hidden_dim // 2, dropout=dropout_rate/2), # Example projection FFN
            RMSNorm(hidden_dim // 2),                 # <<< RMSNorm >>>
            # Maybe one more SwiGLU stage? Or go to final linear?
            nn.Linear(hidden_dim // 2, self.num_classes) # Final layer
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