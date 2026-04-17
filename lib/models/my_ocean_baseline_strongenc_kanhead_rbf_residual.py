import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        g = min(num_groups, out_ch)
        while out_ch % g != 0:
            g -= 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU()

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act(out + identity)
        return out


class StrongEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        c1 = hidden_dim // 2
        c2 = hidden_dim
        c3 = hidden_dim * 2

        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = ResidualConvBlock(c2, c3)

        self.out_channels = c3
        self.skip1_channels = c1
        self.skip2_channels = c2

    def forward(self, x):
        s1 = self.enc1(x)
        x = self.down1(s1)
        s2 = self.enc2(x)
        x = self.down2(s2)
        x = self.enc3(x)
        return x, s1, s2


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, feats):
        tokens = torch.stack(
            [feats[:, t].mean(dim=(2, 3)) for t in range(feats.shape[1])],
            dim=1
        )
        scores = self.score(tokens)
        attn = torch.softmax(scores, dim=1)
        pooled = (feats * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return pooled, attn


class KANHeadRBF(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, num_centers=16, init_gamma=1.0, dropout=0.05):
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


class StrongDecoderKANHeadResidual(nn.Module):
    def __init__(self, bottleneck_channels, skip1_channels, skip2_channels, out_steps=16, out_channels=1,
                 kan_head_hidden_dim=64, kan_head_num_centers=16, kan_head_init_gamma=1.0, kan_head_dropout=0.05):
        super().__init__()
        self.out_steps = out_steps
        self.out_channels = out_channels

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ResidualConvBlock(bottleneck_channels + skip2_channels, bottleneck_channels // 2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ResidualConvBlock(bottleneck_channels // 2 + skip1_channels, bottleneck_channels // 4)

        feat_channels = bottleneck_channels // 4
        self.base_head = nn.Conv2d(feat_channels, out_steps * out_channels, kernel_size=1)
        self.kan_head = KANHeadRBF(
            in_channels=feat_channels,
            out_channels=out_steps * out_channels,
            hidden_dim=kan_head_hidden_dim,
            num_centers=kan_head_num_centers,
            init_gamma=kan_head_init_gamma,
            dropout=kan_head_dropout,
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, skip1, skip2):
        x = self.up1(x)
        diff_h = skip2.size(2) - x.size(2)
        diff_w = skip2.size(3) - x.size(3)
        x = F.pad(x, [0, diff_w, 0, diff_h])
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        diff_h = skip1.size(2) - x.size(2)
        diff_w = skip1.size(3) - x.size(3)
        x = F.pad(x, [0, diff_w, 0, diff_h])
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2(x)

        out_base = self.base_head(x)
        out_kan = self.kan_head(x)
        alpha = torch.sigmoid(self.alpha)
        return out_base + alpha * out_kan


class MY_OCEAN_BASELINE_STRONGENC_KANHEAD_RBF_RESIDUAL(nn.Module):
    def __init__(self, in_shape, hid_S=32, aft_seq_length=16,
                 kan_head_hidden_dim=64, kan_head_num_centers=16,
                 kan_head_init_gamma=1.0, kan_head_dropout=0.05, **kwargs):
        super().__init__()
        self.out_steps = aft_seq_length
        self.out_channels = 1

        self.encoder = StrongEncoder(in_channels=1, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=self.encoder.out_channels)
        self.decoder = StrongDecoderKANHeadResidual(
            bottleneck_channels=self.encoder.out_channels,
            skip1_channels=self.encoder.skip1_channels,
            skip2_channels=self.encoder.skip2_channels,
            out_steps=self.out_steps,
            out_channels=self.out_channels,
            kan_head_hidden_dim=kan_head_hidden_dim,
            kan_head_num_centers=kan_head_num_centers,
            kan_head_init_gamma=kan_head_init_gamma,
            kan_head_dropout=kan_head_dropout
        )

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape

        encoded_seq = []
        last_s1, last_s2 = None, None

        for t in range(T):
            feat, s1, s2 = self.encoder(x_raw[:, t])
            encoded_seq.append(feat)
            last_s1, last_s2 = s1, s2

        encoded_seq = torch.stack(encoded_seq, dim=1)
        pooled, attn = self.temporal(encoded_seq)

        out = self.decoder(pooled, last_s1, last_s2)
        out = out.view(B, self.out_steps, self.out_channels, H, W)
        return out
