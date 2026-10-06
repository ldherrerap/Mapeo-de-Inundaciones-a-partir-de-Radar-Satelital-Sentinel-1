"""
U-Net implementada desde cero en PyTorch (sin bibliotecas de segmentación).

Opciones para las ablaciones:
  depth      : número de niveles de submuestreo (3, 4 o 5)
  base_ch    : canales del primer nivel (se duplican en cada nivel)
  skip_mode  : "concat"    -> U-Net clásica (Ronneberger et al., 2015)
               "attention" -> Attention U-Net (Oktay et al., 2018): compuertas de atención en los saltos
               "none"      -> sin conexiones de salto (encoder-decoder puro, ablación)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BN -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Compuerta de atención aditiva: pondera el mapa del encoder (x) usando la señal
    del decoder (g) como contexto. Devuelve x * alpha, con alpha ∈ [0, 1] por píxel."""

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.w_x = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.last_alpha = None  # se guarda para visualizar en el informe

    def forward(self, g, x):
        a = F.relu(self.w_g(g) + self.w_x(x), inplace=True)
        alpha = self.psi(a)
        self.last_alpha = alpha.detach()
        return x * alpha


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, skip_mode: str):
        super().__init__()
        self.skip_mode = skip_mode
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        up_ch = in_ch // 2
        if skip_mode == "attention":
            self.gate = AttentionGate(up_ch, skip_ch, max(skip_ch // 2, 1))
        conv_in = up_ch + (skip_ch if skip_mode != "none" else 0)
        self.conv = DoubleConv(conv_in, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Ajuste por si la entrada no es múltiplo de 2^depth
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.skip_mode == "none":
            return self.conv(x)
        if self.skip_mode == "attention":
            skip = self.gate(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 1, base_ch: int = 32,
                 depth: int = 4, skip_mode: str = "concat", dropout: float = 0.1):
        super().__init__()
        assert skip_mode in {"concat", "attention", "none"}
        chs = [base_ch * 2 ** i for i in range(depth + 1)]  # p.ej. 32,64,128,256,512

        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[i], chs[i + 1],
                                                      dropout if i == depth - 1 else 0.0))
            for i in range(depth)
        )
        self.ups = nn.ModuleList(
            Up(chs[i + 1], chs[i], chs[i], skip_mode) for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(chs[0], num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)  # logits (B, 1, H, W)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for mode in ["concat", "attention", "none"]:
        m = UNet(skip_mode=mode)
        out = m(torch.randn(2, 2, 256, 256))
        print(f"{mode:9s} -> {tuple(out.shape)}  params={count_params(m)/1e6:.2f}M")
