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


class StemPlusEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.GroupNorm(8, 24),
            nn.SiLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.GroupNorm(8, 24),
            nn.SiLU(),
        )
        self.skip_proj = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(24, hidden_dim)

    def forward(self, x):
        x1 = self.stem(x) + self.skip_proj(x)
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
        )
        scores = self.score(pooled_tokens)
        attn = torch.softmax(scores, dim=1)
        pooled = (feats * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return pooled, attn


class KANHeadRBF(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=96, num_centers=40, init_gamma=1.5, dropout=0.05):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        centers = torch.linspace(-1.5, 1.5, num_centers)
        self.register_buffer("centers", centers)

        self.log_gamma = nn.Parameter(torch.log(torch.tensor(init_gamma, dtype=torch.float32)))
        self.rbf_weight = nn.Parameter(torch.randn(hidden_dim, num_centers) * 0.02)
        self.rbf_bias = nn.Parameter(torch.zeros(hidden_dim))

        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        z = self.in_proj(z)
        z = self.norm(z)
        z = torch.tanh(z)

        gamma = torch.exp(self.log_gamma).clamp(min=1e-4, max=100.0)
        diff = z.unsqueeze(-1) - self.centers
        phi = torch.exp(-gamma * diff.pow(2))

        z = torch.einsum("nhk,hk->nh", phi, self.rbf_weight) + self.rbf_bias
        z = F.silu(z)
        z = self.mix(z)
        z = F.silu(z)
        z = self.dropout(z)
        z = self.out(z)

        z = z.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return z


class DecoderKANResidual(nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        skip_dim=24,
        out_steps=16,
        out_channels=1,
        kan_head_hidden_dim=96,
        kan_head_num_centers=40,
        kan_head_init_gamma=1.5,
        kan_head_dropout=0.05,
    ):
        super().__init__()
        total_out = out_steps * out_channels

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(hidden_dim + skip_dim, 32)

        self.base_head = nn.Conv2d(32, total_out, kernel_size=1)
        self.kan_head = KANHeadRBF(
            in_channels=32,
            out_channels=total_out,
            hidden_dim=kan_head_hidden_dim,
            num_centers=kan_head_num_centers,
            init_gamma=kan_head_init_gamma,
            dropout=kan_head_dropout,
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [0, diff_w, 0, diff_h])
        x = torch.cat([x, skip], dim=1)
        x = self.dec1(x)

        out_base = self.base_head(x)
        out_kan = self.kan_head(x)
        alpha = torch.sigmoid(self.alpha)
        return out_base + alpha * out_kan


class MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_STEMPLUS(nn.Module):
    def __init__(
        self,
        in_shape,
        hid_S=32,
        aft_seq_length=16,
        kan_head_hidden_dim=96,
        kan_head_num_centers=40,
        kan_head_init_gamma=1.5,
        kan_head_dropout=0.05,
        **kwargs
    ):
        super().__init__()
        self.out_steps = aft_seq_length
        self.out_channels = 1

        self.encoder = StemPlusEncoder(in_channels=3, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=hid_S)
        self.decoder = DecoderKANResidual(
            hidden_dim=hid_S,
            skip_dim=24,
            out_steps=self.out_steps,
            out_channels=self.out_channels,
            kan_head_hidden_dim=kan_head_hidden_dim,
            kan_head_num_centers=kan_head_num_centers,
            kan_head_init_gamma=kan_head_init_gamma,
            kan_head_dropout=kan_head_dropout,
        )

    @staticmethod
    def _laplacian(x):
        lap = torch.zeros_like(x)
        lap[:, :, 1:-1, 1:-1] = (
            x[:, :, 2:, 1:-1] + x[:, :, :-2, 1:-1] +
            x[:, :, 1:-1, 2:] + x[:, :, 1:-1, :-2] -
            4.0 * x[:, :, 1:-1, 1:-1]
        )
        return lap

    def _build_sde_features(self, x_raw):
        feats = []
        prev = None
        for t in range(x_raw.shape[1]):
            sst = x_raw[:, t]
            drift = torch.zeros_like(sst) if prev is None else sst - prev
            lap = self._laplacian(sst)
            feats.append(torch.cat([sst, drift, lap], dim=1))
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
        pooled, _ = self.temporal(encoded_seq)

        out = self.decoder(pooled, last_skip)
        return out.view(B, self.out_steps, self.out_channels, H, W)
