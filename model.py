import torch
import torch.nn as nn
import torch.nn.functional as F # Needed if we use F.dropout etc.


# ResBlock definition remains the same
class ResBlock(nn.Module):
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


# --- PredictorModel with Attention ---
# v1.2: Added MultiheadAttention layer before MLP head
class PredictorModel(nn.Module):
    """Predictor/Classifier head with optional attention pre-processing."""
    def __init__(self, features=768, outputs=1, hidden=1024, num_attn_heads=8, attn_dropout=0.1): # Added attention params
        super().__init__()
        self.features = features
        self.outputs = outputs
        self.hidden = hidden

        # --- Attention Layer ---
        # Input features must be divisible by num_heads
        # Adjust num_heads if features=1152 isn't divisible by 8 (1152 / 8 = 144, so it's okay)
        # Or choose num_heads that divide features (e.g., 4, 6, 12)
        actual_num_heads = num_attn_heads
        if features % num_attn_heads != 0:
             # Find a divisor close to the requested num_heads, e.g. 6 or 12 for 1152
             possible_heads = [h for h in [1, 2, 3, 4, 6, 8, 12, 16] if features % h == 0]
             # Get closest possible head count to requested
             actual_num_heads = min(possible_heads, key=lambda x:abs(x-num_attn_heads))
             print(f"Warning: Features ({features}) not divisible by num_attn_heads ({num_attn_heads}). Using {actual_num_heads} heads instead.")

        self.attention = nn.MultiheadAttention(
            embed_dim=self.features,
            num_heads=actual_num_heads,
            dropout=attn_dropout,
            batch_first=True # Important: expects [Batch, Seq, Features]
        )
        # LayerNorm after attention (common practice)
        self.norm_attn = nn.LayerNorm(self.features)
        # --- End Attention Layer ---

        # MLP Head (up block now takes features dimension)
        self.up = nn.Sequential(
            nn.Linear(self.features, self.hidden), # Input is still features dim
            nn.LayerNorm(self.hidden), nn.GELU(),
            ResBlock(ch=self.hidden),
        )
        # Down block remains the same
        self.down = nn.Sequential(
            nn.Linear(self.hidden, 128), nn.LayerNorm(128), nn.GELU(),
            # Use F.dropout here if needed, or keep the layer
            nn.Linear(128, 64), nn.LayerNorm(64), nn.Dropout(0.1), nn.GELU(), # Keep original dropout for now
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, self.outputs),
        )
        # Output activation remains the same
        self.out = nn.Tanh() if self.outputs == 1 else nn.Softmax(dim=1)

    # v1.2.1: Modified forward pass to include attention
    def forward(self, x):
        # x original shape: [Batch, Features]

        # --- Apply Attention ---
        # Add sequence dimension: [Batch, SeqLen=1, Features]
        x_seq = x.unsqueeze(1)
        # Self-attention: query=x, key=x, value=x
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        # Residual connection + LayerNorm
        # attn_output shape is [Batch, 1, Features], remove seq dim before adding
        x_attended = self.norm_attn(x + attn_output.squeeze(1))
        # --- End Apply Attention ---

        # Feed attended features into the MLP head
        y = self.up(x_attended) # Pass attended features to MLP
        z = self.down(y)

        # Apply final activation
        if self.outputs > 1:
            return self.out(z) # Softmax for Classifier
        else:
            tanh_output = self.out(z) # Tanh for Scorer
            return (tanh_output + 1.0) / 2.0 # Scale Tanh [-1, 1] to [0, 1]