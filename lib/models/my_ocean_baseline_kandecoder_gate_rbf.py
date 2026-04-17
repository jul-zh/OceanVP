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


class ChannelKANGateRBF(nn.Module):
    def __init__(self, channels, hidden_dim=64, num_centers=16, init_gamma=1.0, dropout=0.05):
        super().__init__()
        self.in_proj = nn.Linear(channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        centers = torch.linspace(-1.5, 1.5, num_centers)
        self.register_buffer("centers", centers)

        self.log_gamma = nn.Parameter(torch.log(torch.tensor(init_gamma, dtype=torch.float32)))
        self.rbf_weight = nn.Parameter(torch.randn(hidden_dim, num_centers) * 0.02)
        self.rbf_bias = nn.Parameter(torch.zeros(hidden_dim))

        self.mix = nn.Linear(hidden_dim, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,C,H,W]
        pooled = x.mean(dim=(2, 3))  # [B,C]
        z = self.in_proj(pooled)
        z = self.norm(z)
        z = torch.tanh(z)

        gamma = torch.exp(self.log_gamma).clamp(min=1e-4, max=100.0)
        diff = z.unsqueeze(-1) - self.centers
        phi = torch.exp(-gamma * diff.pow(2))
        z = torch.einsum("bhk,hk->bh", phi, self.rbf_weight) + self.rbf_bias
        z = F.silu(z)
        z = self.mix(z)
        z = self.dropout(z)

        gate = torch.sigmoid(z).unsqueeze(-1).unsqueeze(-1)
        return x * gate


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


class MY_OCEAN_BASELINE_KANDECODER_GATE_RBF(nn.Module):
    def __init__(
        self,
        in_shape,
        hid_S=32,
        out_steps=16,
        out_channels=1,
        kan_gate_hidden_dim=64,
        kan_gate_num_centers=16,
        kan_gate_init_gamma=1.0,
        kan_gate_dropout=0.05,
        **kwargs,
    ):
        super().__init__()
        in_channels = in_shape[1]
        self.out_steps = out_steps
        self.out_channels = out_channels

        self.encoder = SmallEncoder(in_channels=in_channels, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=hid_S)
        self.kan_gate = ChannelKANGateRBF(
            channels=hid_S,
            hidden_dim=kan_gate_hidden_dim,
            num_centers=kan_gate_num_centers,
            init_gamma=kan_gate_init_gamma,
            dropout=kan_gate_dropout,
        )
        self.decoder = SmallDecoder(hidden_dim=hid_S, out_steps=out_steps, out_channels=out_channels)

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
        pooled = self.kan_gate(pooled)

        out = self.decoder(pooled, last_skip)
        out = out.view(B, self.out_steps, self.out_channels, H, W)
        return out
