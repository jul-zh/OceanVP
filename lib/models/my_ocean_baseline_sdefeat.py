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
    def __init__(self, in_channels=5, hidden_dim=32):
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
        pooled_tokens = torch.stack(
            [feats[:, t].mean(dim=(2, 3)) for t in range(feats.shape[1])],
            dim=1
        )  # [B,T,C]

        scores = self.score(pooled_tokens)   # [B,T,1]
        attn = torch.softmax(scores, dim=1)
        pooled = (feats * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return pooled, attn


class SmallDecoder(nn.Module):
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


class MY_OCEAN_BASELINE_SDEFEAT(nn.Module):
    def __init__(self, in_shape, hid_S=32, aft_seq_length=16, **kwargs):
        super().__init__()
        self.out_steps = aft_seq_length
        self.out_channels = 1

        self.encoder = SmallEncoder(in_channels=5, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=hid_S)
        self.decoder = SmallDecoder(
            hidden_dim=hid_S,
            out_steps=self.out_steps,
            out_channels=self.out_channels
        )

    @staticmethod
    def _dx(x):
        # x: [B,1,H,W]
        dx = torch.zeros_like(x)
        dx[:, :, :, 1:-1] = 0.5 * (x[:, :, :, 2:] - x[:, :, :, :-2])
        dx[:, :, :, 0] = x[:, :, :, 1] - x[:, :, :, 0]
        dx[:, :, :, -1] = x[:, :, :, -1] - x[:, :, :, -2]
        return dx

    @staticmethod
    def _dy(x):
        dy = torch.zeros_like(x)
        dy[:, :, 1:-1, :] = 0.5 * (x[:, :, 2:, :] - x[:, :, :-2, :])
        dy[:, :, 0, :] = x[:, :, 1, :] - x[:, :, 0, :]
        dy[:, :, -1, :] = x[:, :, -1, :] - x[:, :, -2, :]
        return dy

    @staticmethod
    def _laplacian(x):
        # 5-point stencil
        lap = torch.zeros_like(x)
        lap[:, :, 1:-1, 1:-1] = (
            x[:, :, 2:, 1:-1] +
            x[:, :, :-2, 1:-1] +
            x[:, :, 1:-1, 2:] +
            x[:, :, 1:-1, :-2] -
            4.0 * x[:, :, 1:-1, 1:-1]
        )
        return lap

    def _build_sde_features(self, x_raw):
        # x_raw: [B,T,1,H,W]
        B, T, C, H, W = x_raw.shape
        feats = []

        prev = None
        for t in range(T):
            sst = x_raw[:, t]  # [B,1,H,W]

            if prev is None:
                drift = torch.zeros_like(sst)
            else:
                drift = sst - prev

            lap = self._laplacian(sst)
            abs_lap = lap.abs()

            dx = self._dx(sst)
            dy = self._dy(sst)
            grad_mag = torch.sqrt(dx * dx + dy * dy + 1e-8)

            feat_t = torch.cat([sst, drift, lap, abs_lap, grad_mag], dim=1)  # [B,5,H,W]
            feats.append(feat_t)
            prev = sst

        return torch.stack(feats, dim=1)  # [B,T,5,H,W]

    def forward(self, x_raw, **kwargs):
        # x_raw: [B,T,1,H,W]
        B, T, C, H, W = x_raw.shape

        x_feat = self._build_sde_features(x_raw)  # [B,T,5,H,W]

        encoded_seq = []
        last_skip = None

        for t in range(T):
            feat, skip = self.encoder(x_feat[:, t])
            encoded_seq.append(feat)
            last_skip = skip

        encoded_seq = torch.stack(encoded_seq, dim=1)  # [B,T,C',H/2,W/2]
        pooled, attn = self.temporal(encoded_seq)

        out = self.decoder(pooled, last_skip)  # [B, out_steps, H, W]
        out = out.view(B, self.out_steps, self.out_channels, H, W)
        return out
