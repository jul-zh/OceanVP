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
    def __init__(self, in_channels=2, hidden_dim=32):
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
        )
        scores = self.score(pooled_tokens)
        attn = torch.softmax(scores, dim=1)
        pooled = (feats * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return pooled, attn


class KANHeadSpline(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=96, dropout=0.05):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.base = nn.Linear(hidden_dim, hidden_dim)
        self.spline_u = nn.Linear(hidden_dim, hidden_dim)
        self.spline_v = nn.Linear(hidden_dim, hidden_dim)

        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        z = self.in_proj(z)
        z = self.norm(z)

        base = F.silu(self.base(z))
        spline = torch.sin(self.spline_u(z)) * torch.tanh(self.spline_v(z))
        z = base + spline

        z = self.mix(z)
        z = F.silu(z)
        z = self.dropout(z)
        z = self.out(z)

        z = z.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return z


class SmallDecoderSplineResidual(nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        out_steps=16,
        out_channels=2,
        kan_head_hidden_dim=96,
        kan_head_dropout=0.05,
    ):
        super().__init__()
        total_out = out_steps * out_channels

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(hidden_dim + 16, 32)

        self.base_head = nn.Conv2d(32, total_out, kernel_size=1)
        self.kan_head = KANHeadSpline(
            in_channels=32,
            out_channels=total_out,
            hidden_dim=kan_head_hidden_dim,
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
        out_spline = self.kan_head(x)
        alpha = torch.sigmoid(self.alpha)
        return out_base + alpha * out_spline


class MY_OCEAN_BASELINE_KANHEAD_SPLINE_RESIDUAL_UV(nn.Module):
    def __init__(
        self,
        in_shape,
        hid_S=32,
        aft_seq_length=16,
        kan_head_hidden_dim=96,
        kan_head_dropout=0.05,
        **kwargs
    ):
        super().__init__()
        self.out_steps = aft_seq_length
        self.out_channels = 2

        self.encoder = SmallEncoder(in_channels=2, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=hid_S)
        self.decoder = SmallDecoderSplineResidual(
            hidden_dim=hid_S,
            out_steps=self.out_steps,
            out_channels=self.out_channels,
            kan_head_hidden_dim=kan_head_hidden_dim,
            kan_head_dropout=kan_head_dropout,
        )

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape

        encoded_seq = []
        last_skip = None
        for t in range(T):
            feat, skip = self.encoder(x_raw[:, t])
            encoded_seq.append(feat)
            last_skip = skip

        encoded_seq = torch.stack(encoded_seq, dim=1)
        pooled, _ = self.temporal(encoded_seq)

        out = self.decoder(pooled, last_skip)
        out = out.view(B, self.out_steps, self.out_channels, H, W)
        return out
