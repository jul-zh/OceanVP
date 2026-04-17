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
            nn.Tanh(),
            nn.Linear(dim, 1)
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


class PiecewiseLinearSplineBasis(nn.Module):
    def __init__(self, num_knots=16, xmin=-2.0, xmax=2.0):
        super().__init__()
        knots = torch.linspace(xmin, xmax, num_knots)
        self.register_buffer("knots", knots)
        self.delta = (xmax - xmin) / (num_knots - 1)

    def forward(self, x):
        diff = torch.abs(x.unsqueeze(-1) - self.knots)
        basis = torch.clamp(1.0 - diff / self.delta, min=0.0)
        return basis


class KANHeadSpline(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=64,
        num_knots=16,
        xmin=-2.0,
        xmax=2.0,
        dropout=0.05,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.spline = PiecewiseLinearSplineBasis(
            num_knots=num_knots,
            xmin=xmin,
            xmax=xmax,
        )

        self.spline_weight = nn.Parameter(torch.randn(hidden_dim, num_knots) * 0.02)
        self.spline_bias = nn.Parameter(torch.zeros(hidden_dim))

        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        z = self.in_proj(z)
        z = self.norm(z)
        z = torch.tanh(z)

        phi = self.spline(z)  # [N,Hid,K]
        z = torch.einsum("nhk,hk->nh", phi, self.spline_weight) + self.spline_bias
        z = F.silu(z)
        z = self.mix(z)
        z = F.silu(z)
        z = self.dropout(z)
        z = self.out(z)

        z = z.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return z


class SmallDecoderKANHeadSplineResidual(nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        out_steps=16,
        out_channels=1,
        kan_head_hidden_dim=64,
        kan_head_num_knots=16,
        kan_head_xmin=-2.0,
        kan_head_xmax=2.0,
        kan_head_dropout=0.05,
    ):
        super().__init__()
        self.out_steps = out_steps
        self.out_channels = out_channels

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(hidden_dim + 16, 32)

        self.base_head = nn.Conv2d(32, out_steps * out_channels, kernel_size=1)
        self.spline_head = KANHeadSpline(
            in_channels=32,
            out_channels=out_steps * out_channels,
            hidden_dim=kan_head_hidden_dim,
            num_knots=kan_head_num_knots,
            xmin=kan_head_xmin,
            xmax=kan_head_xmax,
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
        out_spline = self.spline_head(x)

        alpha = torch.sigmoid(self.alpha)
        return out_base + alpha * out_spline


class MY_OCEAN_BASELINE_KANHEAD_SPLINE_RESIDUAL(nn.Module):
    def __init__(
        self,
        in_shape,
        hid_S=32,
        out_steps=16,
        out_channels=1,
        kan_head_hidden_dim=64,
        kan_head_num_knots=16,
        kan_head_xmin=-2.0,
        kan_head_xmax=2.0,
        kan_head_dropout=0.05,
        **kwargs,
    ):
        super().__init__()
        in_channels = in_shape[1]
        self.out_steps = out_steps
        self.out_channels = out_channels

        self.encoder = SmallEncoder(in_channels=in_channels, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=hid_S)
        self.decoder = SmallDecoderKANHeadSplineResidual(
            hidden_dim=hid_S,
            out_steps=out_steps,
            out_channels=out_channels,
            kan_head_hidden_dim=kan_head_hidden_dim,
            kan_head_num_knots=kan_head_num_knots,
            kan_head_xmin=kan_head_xmin,
            kan_head_xmax=kan_head_xmax,
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
        pooled, attn = self.temporal(encoded_seq)

        out = self.decoder(pooled, last_skip)
        out = out.view(B, self.out_steps, self.out_channels, H, W)
        return out
