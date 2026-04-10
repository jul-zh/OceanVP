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


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

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


class MeanDecoder(nn.Module):
    def __init__(self, hidden_dim=32, out_steps=16, out_channels=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = ConvBlock(hidden_dim + 16, 32)
        self.out = nn.Conv2d(32, out_steps * out_channels, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [0, diff_w, 0, diff_h])
        x = torch.cat([x, skip], dim=1)
        feat = self.dec(x)
        out = self.out(feat)
        return out, feat


class RBFKANHeadV2(nn.Module):
    """
    Более маленький и мягкий KAN-like pixelwise sigma head.
    Вход:  [B, C, H, W]
    Выход: [B, out_ch, H, W]
    """
    def __init__(self, in_ch, out_ch, hidden=4, num_basis=6):
        super().__init__()
        self.hidden = hidden
        self.num_basis = num_basis

        self.in_proj = nn.Conv2d(in_ch, hidden, kernel_size=1)

        base_centers = torch.linspace(-1.5, 1.5, num_basis).view(1, 1, num_basis, 1, 1)
        base_centers = base_centers.repeat(1, hidden, 1, 1, 1)
        self.centers = nn.Parameter(base_centers)

        # начальные ширины не слишком маленькие
        self.log_widths = nn.Parameter(torch.full((1, hidden, num_basis, 1, 1), -0.2))

        self.out_weight = nn.Parameter(torch.randn(out_ch, hidden, num_basis) * 0.03)
        self.out_bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        # x: [B, C, H, W]
        z = self.in_proj(x)   # [B, hidden, H, W]
        z = z.unsqueeze(2)    # [B, hidden, 1, H, W]

        widths = F.softplus(self.log_widths) + 1e-3
        phi = torch.exp(-((z - self.centers) / widths) ** 2)  # [B, hidden, basis, H, W]

        out = torch.einsum('bckhw,ock->bohw', phi, self.out_weight)
        out = out + self.out_bias.view(1, -1, 1, 1)
        return out


class PROBKANV2_Model(nn.Module):
    def __init__(self, in_shape, hid_S=32, aft_seq_length=16, **kwargs):
        super().__init__()
        self.out_steps = aft_seq_length
        self.out_channels = 1

        self.encoder = SmallEncoder(in_channels=1, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=hid_S)

        self.mean_decoder = MeanDecoder(
            hidden_dim=hid_S,
            out_steps=self.out_steps,
            out_channels=self.out_channels
        )

        self.sigma_kan = RBFKANHeadV2(
            in_ch=32,
            out_ch=self.out_steps * self.out_channels,
            hidden=4,
            num_basis=6
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

        encoded_seq = torch.stack(encoded_seq, dim=1)  # [B, T, C, H', W']
        pooled, attn = self.temporal(encoded_seq)

        mu, mean_feat = self.mean_decoder(pooled, last_skip)
        raw_log_sigma = self.sigma_kan(mean_feat)

        mu = mu.view(B, self.out_steps, self.out_channels, H, W)
        raw_log_sigma = raw_log_sigma.view(B, self.out_steps, self.out_channels, H, W)
        return mu, raw_log_sigma
