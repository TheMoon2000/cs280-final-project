import math
import torch
import torch.nn as nn

class UformerSimple(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.dropout_rate = dropout
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.encoder1 = nn.Sequential(
            WindowAttention(8, downsample=False, in_channels=32, dropout=dropout),
            WindowAttention(8, downsample=True, in_channels=32, dropout=dropout)
        )
        self.encoder2 = nn.Sequential(
            WindowAttention(8, downsample=False, in_channels=64, dropout=dropout),
            WindowAttention(8, downsample=True, in_channels=64, dropout=dropout)
        )
        self.encoder3 = nn.Sequential(
            WindowAttention(8, downsample=False, in_channels=128, num_heads=4, dropout=dropout),
            WindowAttention(8, downsample=True, in_channels=128, num_heads=4, dropout=dropout)
        )
        self.bottleneck =  WindowAttention(7, downsample=False, in_channels=256, num_heads=8, dropout=dropout)

        self.upsample1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder1 = WindowAttention(8, downsample=False, in_channels=256, num_heads=4, dropout=dropout)
        self.upsample2 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.decoder2 = WindowAttention(8, downsample=False, in_channels=128, dropout=dropout)
        self.upsample3 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        self.decoder3 = WindowAttention(8, downsample=False, in_channels=64, dropout=dropout)
        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor):
        delta = self.input_conv(x)
        delta = torch.relu(delta)

        delta, delta2 = self.encoder1(delta) # 32, 64 channels
        delta2, delta3 = self.encoder2(delta2) # 64, 128 channels
        delta3, delta4 = self.encoder3(delta3) # 128, 256 channels
        delta4 = self.bottleneck(delta4)
        delta4 = self.upsample1(delta4) # 128 channels
        delta3 = torch.cat([delta4, delta3], dim=1) # 128 + 128 channels
        delta3 = self.decoder1(delta3)
        delta3 = self.upsample2(delta3) # 256 -> 64 channels
        delta2 = torch.cat([delta3, delta2], dim=1) # 64 + 64 channels
        delta2 = self.decoder2(delta2)
        delta2 = self.upsample3(delta2)
        delta = torch.cat([delta2, delta], dim=1) # 32 + 32 channels
        delta = self.decoder3(delta)

        delta = self.output_conv(delta)
        
        return x + delta

class WindowAttention(nn.Module):
    def __init__(self, window_size=8, downsample=False, in_channels=32, num_layers=2, num_heads=2, dropout=0.1):
        super().__init__()

        self.W = window_size
        self.C = in_channels
        self.downsample = downsample
        self.norm1 = nn.LayerNorm(in_channels)
        self.window = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads, dim_feedforward=in_channels * 2, batch_first=True), num_layers=num_layers)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(in_channels)
        self.linear1 = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, groups=in_channels * 2, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.linear2 = nn.Linear(in_channels * 2, in_channels)
        
        if downsample:
            self.out_layer = nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2)
        else:
            self.out_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Embedding table
        # self.bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        # coords_h = torch.arange(window_size) # [0,...,Wh-1]
        # coords_w = torch.arange(window_size) # [0,...,Ww-1]
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # coords_flatten = torch.flatten(coords, 1)
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        # relative_coords[:, :, 1] += window_size - 1
        # relative_coords[:, :, 0] *= window_size * window_size - 1
        # self.relative_position_index = relative_coords.sum(-1)
        self.positional_embedding = nn.Parameter(torch.zeros((1, window_size ** 2, in_channels)))
        nn.init.kaiming_normal_(self.positional_embedding.data)

    def forward(self, x: torch.Tensor):
        x_ = x
        x = self.norm1(x.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
        x = x.unfold(2, self.W, self.W).unfold(3, self.W, self.W) # (B, C, Ph, Pw, Wh, Ww)
        Ph, Pw = x.shape[2], x.shape[3]
        x = x.permute((0, 2, 3, 4, 5, 1)).flatten(0, 2).flatten(1, 2) # (B * Ph * Pw, Wh * Ww, C)
        
        # Add positional embedding
        x += self.positional_embedding

        x = self.window(x)
        x = x.view(x_.shape[0], Ph, Pw, self.W, self.W, self.C).permute(0, 5, 1, 3, 2, 4) # (B, C, Ph, Wh, Pw, Ww)
        x = x.flatten(2, 3).flatten(3, 4)
        x = self.dropout1(x)

        x += x_
        x_ = x
        x = self.norm2(x.permute(0, 2, 3, 1)) # (B, H, W, C)
        x = self.linear1(x)
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.linear2(x)
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        x = self.dropout2(x)
        x += x_

        if self.downsample:
            return x, self.out_layer(x)
        else:
            x = self.out_layer(x)
                
        return x

# Attention over patches. Didn't seem to work well
class Block(nn.Module):
    def __init__(self, window_size=8, downsample=False, in_channels=32, num_layers=2):
        super().__init__()

        self.W = window_size
        self.in_channels = in_channels
        self.downsample = downsample
        self.norm1 = nn.LayerNorm(in_channels)
        self.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=in_channels * window_size ** 2, nhead=2, batch_first=True), num_layers)
        if downsample:
            self.out_layer = nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2)
        else:
            self.out_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.position_embedding = nn.Parameter(torch.zeros((window_size * 2 - 1, window_size * 2 - 1)))
        nn.init.kaiming_normal_(self.position_embedding.data)
    
    def forward(self, delta: torch.Tensor):
        C = delta.shape[1]
        x = delta
        delta = self.norm1(delta.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))

        height_padding = (self.W - delta.shape[2] % self.W) % self.W
        width_padding = (self.W - delta.shape[3] % self.W) % self.W
        if height_padding:
            delta = torch.cat([delta, torch.zeros((delta.shape[0], C, height_padding, delta.shape[3]), device=delta.device)], dim=2)
        if width_padding:
            delta = torch.cat([delta, torch.zeros((delta.shape[0], C, delta.shape[2], width_padding), device=delta.device)], dim=3)

        patches = delta.unfold(2, self.W, self.W).unfold(3, self.W, self.W)
        window_width = patches.shape[3]
        patches = patches.flatten(2, 3).transpose(1, 2).flatten(2, 4) # (B, L, CHW)
        layer1 = self.t(patches)
        delta = layer1.unfold(1, window_width, window_width).unfold(2, C, C).unfold(2, self.W, self.W) # (B, Ph, Wh, Pw, C, Ww)
        delta = delta.permute((0, 4, 1, 2, 3, 5)).flatten(4, 5).flatten(2, 3)

        # Undo padding
        if height_padding:
            delta = delta[:, :, :-height_padding]
        if width_padding:
            delta = delta[:, :, :, :-width_padding]
        
        delta += x

        if self.downsample:
            return delta, self.out_layer(delta)
        else:
            delta = self.out_layer(delta)
        
        return delta


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