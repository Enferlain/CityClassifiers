# model.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    # ... (ResBlock remains the same) ...
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch) # Maybe LayerNorm is better here than BatchNorm? Let's try LN first.
        self.long = nn.Sequential(
            nn.Linear(ch, ch),
            nn.GELU(), # Changed to GELU from LeakyReLU, often works well with Norms
            nn.Linear(ch, ch),
            nn.GELU(), # Changed to GELU
            nn.Linear(ch, ch),
        )
    def forward(self, x):
        # Apply LayerNorm before residual connection - Pre-Norm style
        return x + self.long(self.norm(x))


class PredictorModel(nn.Module):
    """Main predictor class"""
    # v1.1: Added LayerNorm/BatchNorm and GELU
    def __init__(self, features=768, outputs=1, hidden=1024):
        super().__init__()
        self.features = features
        self.outputs = outputs
        self.hidden = hidden
        # Consider LayerNorm or BatchNorm in the up block too
        self.up = nn.Sequential(
            nn.Linear(self.features, self.hidden),
            # nn.BatchNorm1d(self.hidden), # Option 1: BatchNorm
            nn.LayerNorm(self.hidden),     # Option 2: LayerNorm (often better for transformer-like features)
            nn.GELU(),                     # Often pairs well with Norm layers
            ResBlock(ch=self.hidden),
        )
        self.down = nn.Sequential(
            # Maybe another Norm here?
            # nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, 128),
            nn.LayerNorm(128), # Added LayerNorm
            nn.GELU(),         # Changed to GELU
            nn.Linear(128, 64),
            nn.LayerNorm(64),  # Added LayerNorm
            nn.Dropout(0.1),   # Keep dropout
            nn.GELU(),         # Changed to GELU
            nn.Linear(64, 32),
            nn.LayerNorm(32),  # Added LayerNorm right before the final linear layer
            nn.GELU(),         # Changed to GELU
            nn.Linear(32, self.outputs), # Final layer to produce pre-activation value
        )
        # Keep Tanh for [0,1] output for now, relying on Norm layers to control input range
        self.out = nn.Tanh() if self.outputs == 1 else nn.Softmax(dim=1)

    # v1.1.1: Removed debug print
    def forward(self, x):
        y = self.up(x)
        z = self.down(y)
        # ---- DEBUG print removed ----
        if self.outputs > 1:
            return self.out(z) # Softmax
        else:
            # This path isn't used for arch: class, but kept for consistency
            tanh_output = self.out(z)
            return (tanh_output + 1.0) / 2.0 # Tanh -> [0, 1]