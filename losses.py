import torch
import torch.nn as nn
import torch.nn.functional as F

class GHMC_Loss(nn.Module):
    """ Gradient Harmonizing Mechanism for Classification (GHM-C)
        Adapts loss weights based on the distribution of example difficulty (approximated by gradient norm magnitude).
        Uses CrossEntropy as the base loss.

        Args:
            bins (int): Number of bins to approximate the gradient norm distribution.
            momentum (float): Momentum factor for updating the bin counts (EMA).
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    def __init__(self, bins=10, momentum=0.75, reduction='mean'):
        super(GHMC_Loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins # Bin edges [0, 0.1, 0.2, ..., 1.0]
        self.reduction = reduction

        # Register buffers for bin counts and last average counts (for EMA)
        # We need these on the correct device eventually, but register first
        self.register_buffer('acc_sum', torch.zeros(bins))
        self.register_buffer('last_acc_sum', torch.zeros(bins))
        self.is_initialized = False # Flag to handle first iteration

        print(f"DEBUG GHMC_Loss Init: bins={bins}, momentum={momentum}, reduction='{reduction}'")

    def _get_grad_norm_approx(self, logits, targets):
        """ Approximates gradient norm magnitude using prediction probability """
        probs = torch.sigmoid(logits) if logits.shape[-1] == 1 else F.softmax(logits, dim=-1)
        # Get probability of the TRUE class
        if probs.shape[-1] == 1: # Binary case with single logit
             # targets should be 0.0 or 1.0 float
             p_target = probs * targets + (1 - probs) * (1 - targets)
        else: # Multi-class case with CE-like targets
             # targets should be Long class indices
             p_target = probs.gather(1, targets.view(-1, 1)).view(-1)

        # Gradient norm proxy 'g': Larger 'g' means harder example (probability further from 1.0)
        g = torch.abs(p_target - 1.0)
        return g

    def forward(self, logits, targets):
        """ Calculate GHM-C loss.

        Args:
            logits (Tensor): Raw model output logits. Shape [N, C] for C classes, or [N] or [N, 1] for binary (BCE-style).
            targets (Tensor): Ground truth labels. Shape [N] (Long indices for CE-style) or [N] (Float 0/1 for BCE-style).

        Returns:
            Tensor: Calculated GHM-C loss.
        """
        num_classes = logits.shape[-1]
        if num_classes == 1 and self.acc_sum.device != logits.device:
             # Special handling maybe needed if we treat BCE-style as multi-class internally?
             # Let's assume CE-style input/target format for simplicity first.
             print(f"Warning: GHMC_Loss might work best with num_classes >= 2 and CE-style targets.")
             # Or adapt the grad norm calculation for single logit output?

        if num_classes <= 1 and logits.ndim > 1: logits = logits.squeeze(-1) # Ensure [N] if needed

        # --- Ensure targets are Long for gather() in _get_grad_norm_approx ---
        # This means for binary case, we expect 0 or 1 integer labels
        if targets.dtype != torch.long:
             try: targets_long = targets.long()
             except: raise TypeError(f"GHMC_Loss requires Long targets for gather, got {targets.dtype}")
        else: targets_long = targets
        # --- End Target Type Check ---

        # Move edges to the correct device if needed
        if self.edges.device != logits.device:
            self.edges = self.edges.to(logits.device)
            self.acc_sum = self.acc_sum.to(logits.device) # Ensure buffer is on correct device

        # Calculate gradient norm approximation 'g'
        g = self._get_grad_norm_approx(logits, targets_long)

        # Calculate weights based on 'g' distribution
        weights = torch.zeros_like(logits[:, 0] if logits.ndim > 1 else logits) # Match batch size, device

        # Calculate bin index for each example based on 'g'
        bin_indices = torch.floor(g * self.bins).long()
        # Clamp indices to be within [0, bins-1]
        bin_indices = torch.clamp(bin_indices, 0, self.bins - 1)

        # Count examples in each bin for the current batch
        bin_counts = torch.zeros(self.bins, device=logits.device)
        bin_indices_unique, counts = torch.unique(bin_indices, return_counts=True)
        bin_counts[bin_indices_unique] = counts.float()

        # Update accumulated bin sums using EMA
        if not self.is_initialized: # First batch
            self.acc_sum = bin_counts
            self.is_initialized = True
        else:
            # Store previous acc_sum before update if momentum > 0
            if self.momentum > 0:
                self.last_acc_sum.copy_(self.acc_sum) # Store previous state for momentum adjustment
            # EMA update
            self.acc_sum = self.momentum * self.acc_sum + (1 - self.momentum) * bin_counts

        # Avoid division by zero if a bin is empty
        num_examples = logits.shape[0]
        safe_acc_sum = self.acc_sum.clamp(min=1e-6) # Use accumulated sums
        beta = num_examples / safe_acc_sum # Weight = N / (examples in bin)

        # Calculate weights for each example
        weights = beta[bin_indices]

        # --- Calculate the base loss (Cross Entropy) ---
        # Need to handle binary (1 logit) vs multi-class (C logits)
        if num_classes == 1:
            # Apply sigmoid implicitly, calculate binary cross entropy
            # This part doesn't directly use the weights in standard BCE...
            # GHM usually modifies CE loss, let's stick to CE style input for now.
            # Re-evaluate if BCE input is strictly needed.
            print("Warning/TODO: GHMC_Loss with single logit output needs review.")
            # Fallback to CE calculation assuming logits represent class 1 vs class 0 implicitly?
            # Need to ensure logits are [N, 2] for standard CE or adapt.
            # Let's force use of num_classes=2 for now with GHMC.
            raise NotImplementedError("GHMC_Loss currently expects CE-style input (logits shape [N, C] where C>=2).")

        elif num_classes >= 2:
             # Standard CrossEntropyLoss expects logits [N, C] and targets [N] (Long)
             ce_loss = F.cross_entropy(logits, targets_long, reduction='none')
             # Apply GHM weights
             weighted_loss = ce_loss * weights
        else:
             raise ValueError(f"Invalid num_classes ({num_classes})")

        # Apply reduction
        if self.reduction == 'mean':
            # Normalize by total weight sum or by batch size? Paper suggests batch size.
            # return weighted_loss.sum() / weights.sum().clamp(min=1e-6) # Normalize by weight sum
            return weighted_loss.mean() # Normalize by batch size (seems more common)
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else: # 'none'
            return weighted_loss


# --- Focal Loss Definition ---
# (Keep the FocalLoss class definition here as before)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return torch.mean(focal_loss)
        elif self.reduction == 'sum': return torch.sum(focal_loss)
        else: return focal_loss
# --- End Focal Loss ---
