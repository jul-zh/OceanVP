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
        s1 = self.enc1(x)          # 32x64
        x = self.down1(s1)
        s2 = self.enc2(x)          # 16x32
        x = self.down2(s2)
        x = self.enc3(x)           # 8x16
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
        )  # [B,T,C]

        scores = self.score(tokens)   # [B,T,1]
        attn = torch.softmax(scores, dim=1)
        pooled = (feats * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return pooled, attn


class StrongDecoder(nn.Module):
    def __init__(self, bottleneck_channels, skip1_channels, skip2_channels, out_steps=16, out_channels=1):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ResidualConvBlock(bottleneck_channels + skip2_channels, bottleneck_channels // 2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ResidualConvBlock(bottleneck_channels // 2 + skip1_channels, bottleneck_channels // 4)

        self.out_conv = nn.Conv2d(bottleneck_channels // 4, out_steps * out_channels, kernel_size=1)

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

        return self.out_conv(x)


class MY_OCEAN_BASELINE_STRONGENC(nn.Module):
    def __init__(self, in_shape, hid_S=32, aft_seq_length=16, **kwargs):
        super().__init__()
        self.out_steps = aft_seq_length
        self.out_channels = 1

        self.encoder = StrongEncoder(in_channels=1, hidden_dim=hid_S)
        self.temporal = TemporalAttentionPool(dim=self.encoder.out_channels)
        self.decoder = StrongDecoder(
            bottleneck_channels=self.encoder.out_channels,
            skip1_channels=self.encoder.skip1_channels,
            skip2_channels=self.encoder.skip2_channels,
            out_steps=self.out_steps,
            out_channels=self.out_channels
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
