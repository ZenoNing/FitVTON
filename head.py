import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Apply LayerNorm over channel dimension for (B, C, H, W) tensors."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm stabilization."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = LayerNorm2d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = LayerNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Inject textual context via multi-head cross attention."""

    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_norm = nn.LayerNorm(context_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(context_dim, hidden_dim)
        self.v_proj = nn.Linear(context_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        spatial = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        spatial = self.hidden_norm(spatial)
        context = context.to(spatial.dtype)
        context = self.context_norm(context)
        q = self.q_proj(spatial)
        k = self.k_proj(context)
        v = self.v_proj(context)
        attn_out, _ = self.attn(q, k, v)
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + attn_out


class GarmentMaskHead(nn.Module):
    """Lightweight U-Net style head with textual cross attention."""

    def __init__(
        self,
        in_channels: int,
        out_size,
        hidden_dims=(64, 128, 256),
        num_heads: int = 8,
    ):
        super().__init__()
        self.out_size = out_size

        h1, h2, h3 = hidden_dims

        self.input_norm = nn.BatchNorm2d(in_channels)
        self.enc1 = ResidualBlock(in_channels, h1)
        self.down1 = nn.Sequential(
            nn.Conv2d(h1, h2, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(h2),
            nn.GELU(),
        )
        self.enc2 = ResidualBlock(h2, h2)
        self.down2 = nn.Sequential(
            nn.Conv2d(h2, h3, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(h3),
            nn.GELU(),
        )
        self.enc3 = ResidualBlock(h3, h3)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(h3, h2, kernel_size=2, stride=2),
            LayerNorm2d(h2),
            nn.GELU(),
        )
        self.dec1 = ResidualBlock(h2 + h2, h2)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(h2, h1, kernel_size=2, stride=2),
            LayerNorm2d(h1),
            nn.GELU(),
        )
        self.dec2 = ResidualBlock(h1 + h1, h1)

        self.final_out = nn.Conv2d(h1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, prompt_embeds: torch.Tensor | None = None):
        # x: (B, N, C) or (B, C, H, W)
        if x.dim() == 3:
            b, n, c = x.shape
            h = w = int(n ** 0.5)
            x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)

        # Normalize raw pixel inputs to stabilize training when decoding from VAE
        x = self.input_norm(x)

        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)

        # Prompt-agnostic: keep the module for compatibility, but do not use it.

        u1 = self.up1(e3)
        u1 = torch.cat([u1, e2], dim=1)
        u1 = self.dec1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = self.dec2(u2)

        final_logits = self.final_out(u2)
        # final_logits = F.interpolate(final_logits, size=self.out_size, mode="bilinear", align_corners=False)

        return final_logits
