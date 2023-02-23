import math
import torch
import torch.nn as nn

class MainNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=32*16*16, nhead=2)
        self.t1 = nn.TransformerEncoder(encoder_layer, 1)

    
    def forward(self, x: torch.Tensor):
        delta = self.input_conv(x)
        delta = torch.relu(delta)


        # Size 32 window attention
        W = 16
        C = delta.shape[1]
        # Pad sides to be multiples of 16
        height_padding = (W - x.shape[2] % W) % W
        width_padding = (W - x.shape[3] % W) % W
        if height_padding:
            delta = torch.cat([delta, torch.zeros((x.shape[0], C, height_padding, delta.shape[3]), device=x.device)], dim=2)
        if width_padding:
            delta = torch.cat([delta, torch.zeros((x.shape[0], C, delta.shape[2], width_padding), device=x.device)], dim=3)

        patches = delta.unfold(2, W, W).unfold(3, W, W)
        window_width = patches.shape[3]
        patches = patches.flatten(2, 3).transpose(1, 2).flatten(2, 4) # (B, L, E)
        layer1 = self.t1(patches)
        delta = layer1.unfold(1, window_width, window_width).unfold(2, C, C).unfold(2, W, W) # (B, Wh, Ph, Ww, C, Pw)
        delta = delta.permute((0, 4, 1, 2, 3, 5)).flatten(4, 5).flatten(2, 3)
        
        delta = self.output_conv(delta)

        # Undo padding
        if height_padding:
            delta = delta[:, :, :-height_padding]
        if width_padding:
            delta = delta[:, :, :, :-width_padding]
        return x + delta
    


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss