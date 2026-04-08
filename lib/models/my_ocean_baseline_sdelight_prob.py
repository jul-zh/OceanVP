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
    def __init__(self, in_channels=3, hidden_dim=32):
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


class DualDecoder(nn.Module):
    def __init__(self, hidden_dim=32, out_steps=16, out_channels=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.mean_dec = ConvBlock(hidden_dim + 16, 32)
        self.mean_out = nn.Conv2d(32, out_steps * out_channels, kernel_size=1)

        self.sigma_dec = ConvBlock(hidden_dim + 16, 32)
        self.sigma_out = nn.Conv2d(32, out_steps * out_channels, kernel_size=1)

    def _decode_branch(self, x, skip, dec_block, out_head):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [0, diff_w, 0, diff_h])
        x = torch.cat([x, skip], dim=1)
        x = dec_block(x)
        return out_head(x)

    def forward(self, x, skip):
        mu = self._decode_branch(x, skip, self.mean_dec, self.mean_out)
        raw_sigma = self._decode_branch(x, skip, self.sigma_dec, self.sigma_out)
        return mu, raw_sigma


class MY_OCEAN_BASELINE_SDELIGHT_PROB(nn.Module):
    def __init__(self, in_shape, hid_S=32, aft_seq_length=16, **kwargs):
        super().__init__()
        self.out_steps = aft_seq_length
        self.out_channels = 1

        self.encoder = SmallEncoder(in_channels=3, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=hid_S)
        self.decoder = DualDecoder(
            hidden_dim=hid_S,
            out_steps=self.out_steps,
            out_channels=self.out_channels
        )

    @staticmethod
    def _laplacian(x):
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
        feats = []
        prev = None
        for t in range(x_raw.shape[1]):
            sst = x_raw[:, t]
            if prev is None:
                drift = torch.zeros_like(sst)
            else:
                drift = sst - prev
            lap = self._laplacian(sst)
            feat_t = torch.cat([sst, drift, lap], dim=1)
            feats.append(feat_t)
            prev = sst
        return torch.stack(feats, dim=1)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x_feat = self._build_sde_features(x_raw)

        encoded_seq = []
        last_skip = None
        for t in range(T):
            feat, skip = self.encoder(x_feat[:, t])
            encoded_seq.append(feat)
            last_skip = skip

        encoded_seq = torch.stack(encoded_seq, dim=1)
        pooled, attn = self.temporal(encoded_seq)

        mu, raw_sigma = self.decoder(pooled, last_skip)
        mu = mu.view(B, self.out_steps, self.out_channels, H, W)
        raw_sigma = raw_sigma.view(B, self.out_steps, self.out_channels, H, W)
        return mu, raw_sigma
