import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        g = min(num_groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SmallEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16, hidden_dim)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        return x3, x1


class MeanDecoder(nn.Module):
    def __init__(self, hidden_dim=32, out_steps=16, out_channels=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(hidden_dim + 16, 32)
        self.out_conv = nn.Conv2d(32, out_steps * out_channels, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [0, diff_w, 0, diff_h])
        x = torch.cat([x, skip], dim=1)
        x = self.dec1(x)
        return self.out_conv(x)

class RBFKANScorer(nn.Module):
    """
    KAN-like scorer for temporal attention.
    Input:  [B, T, C]
    Output: [B, T, 1]
    """
    def __init__(self, dim, hidden=8, num_basis=8):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden)

        base_centers = torch.linspace(-1.5, 1.5, num_basis).view(1, 1, 1, num_basis)
        base_centers = base_centers.repeat(1, 1, hidden, 1)   # [1,1,H,K]
        self.centers = nn.Parameter(base_centers)

        self.log_widths = nn.Parameter(torch.full((1, 1, hidden, num_basis), -0.2))
        self.out_weight = nn.Parameter(torch.randn(hidden, num_basis) * 0.05)
        self.out_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, T, C]
        z = self.in_proj(x)              # [B, T, H]
        z = z.unsqueeze(-1)              # [B, T, H, 1]

        widths = F.softplus(self.log_widths) + 1e-3   # [1,1,H,K]
        phi = torch.exp(-((z - self.centers) / widths) ** 2)  # [B,T,H,K]

        score = torch.einsum('bthk,hk->bt', phi, self.out_weight)  # [B,T]
        score = score.unsqueeze(-1) + self.out_bias.view(1, 1, 1)  # [B,T,1]
        return score

class TemporalKANAttentionPool(nn.Module):
    def __init__(self, dim, hidden=8, num_basis=8):
        super().__init__()
        self.score = RBFKANScorer(dim=dim, hidden=hidden, num_basis=num_basis)

    def forward(self, feats):
        # feats: [B, T, C, H, W]
        pooled_tokens = torch.stack(
            [feats[:, t].mean(dim=(2, 3)) for t in range(feats.shape[1])],
            dim=1
        )  # [B, T, C]

        scores = self.score(pooled_tokens)  # [B, T, 1]
        attn = torch.softmax(scores, dim=1)
        pooled = (feats * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return pooled, attn


class KANATTN_Model(nn.Module):
    def __init__(self, in_shape, hid_S=32, aft_seq_length=16, **kwargs):
        super().__init__()
        self.out_steps = aft_seq_length
        self.out_channels = 1

        self.encoder = SmallEncoder(in_channels=1, hidden_dim=hid_S)
        self.temporal = TemporalKANAttentionPool(dim=hid_S, hidden=8, num_basis=8)
        self.decoder = MeanDecoder(
            hidden_dim=hid_S,
            out_steps=self.out_steps,
            out_channels=self.out_channels
        )

    def forward(self, x_raw, **kwargs):
        # x_raw: [B, T, 1, H, W]
        B, T, C, H, W = x_raw.shape

        encoded_seq = []
        last_skip = None

        for t in range(T):
            feat, skip = self.encoder(x_raw[:, t])
            encoded_seq.append(feat)
            last_skip = skip

        encoded_seq = torch.stack(encoded_seq, dim=1)  # [B,T,C,H',W']
        pooled, attn = self.temporal(encoded_seq)

        out = self.decoder(pooled, last_skip)
        out = out.view(B, self.out_steps, self.out_channels, H, W)
        return out
