"""
PlacementDETR — Class-Conditioned Placement DETR with Multi-Scale Feature Fusion.

A ResNet-50-backed DETR model that predicts plausible bounding box placements
for a given object class on a background image.

Architecture:
  - Frozen ResNet-50 backbone (multi-scale: layer3 + layer4)
  - FPN-style feature fusion (C4 + C5 → 256-d)
  - 2D sinusoidal positional encoding
  - Standard Transformer encoder-decoder (6+6 layers, 8 heads)
  - Class-conditioned queries: learned class embedding + query offsets
  - Sigmoid bbox head (corner format [x, y, w, h] in [0, 1])
  - Plausibility head (logit → sigmoid for ranking)
"""
import math

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


IMG_SIZE = 512
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
NUM_QUERIES = 50


def center_crop(img, size=IMG_SIZE):
    """Center crop a PIL image to a square and resize to size x size."""
    w, h = img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img_cropped = img.crop((left, top, left + crop_size, top + crop_size))
    from PIL import Image
    img_resized = img_cropped.resize((size, size), Image.BILINEAR)
    return img_resized


class PositionalEncodingSine(nn.Module):
    """Standard DETR 2D sinusoidal positional encoding."""
    def __init__(self, hidden_dim=256, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x):
        B, _, H, W = x.shape
        device = x.device
        y_embed = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H + eps) * self.scale
            x_embed = x_embed / (W + eps) * self.scale
        dim_t = torch.arange(self.hidden_dim // 2, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 2))
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos = torch.cat([pos_y, pos_x], dim=-1)
        pos = pos.flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
        return pos


class MultiScaleBackbone(nn.Module):
    """Frozen ResNet-50 backbone with multi-scale feature extraction (C4, C5)."""
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        self.stage5 = resnet.layer4
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        c4 = self.stage4(x)
        c5 = self.stage5(c4)
        return {"c4": c4, "c5": c5}


class PlacementDETR(nn.Module):
    """Class-Conditioned Placement DETR with Multi-Scale Feature Fusion."""
    def __init__(self, num_classes, num_queries=100, hidden_dim=256,
                 nheads=8, enc_layers=6, dec_layers=6, dim_feedforward=2048,
                 dropout=0.1, use_cached_features=False):
        super().__init__()
        self.num_queries = num_queries
        self.use_cached_features = use_cached_features
        self.hidden_dim = hidden_dim

        if not use_cached_features:
            self.backbone = MultiScaleBackbone()

        self.proj_c4 = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        self.proj_c5 = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.fusion_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.pos_encoder = PositionalEncodingSine(hidden_dim)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.query_offsets = nn.Parameter(torch.randn(num_queries, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4), nn.Sigmoid())
        self.plausibility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self._init_weights()

    def _init_weights(self):
        for proj in [self.proj_c4, self.proj_c5, self.input_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.fusion_conv.weight)
        nn.init.zeros_(self.fusion_conv.bias)
        nn.init.normal_(self.query_offsets, std=1.0)

    def _fuse_multiscale(self, c4, c5):
        p4 = self.proj_c4(c4)
        p5 = self.proj_c5(c5)
        p5_up = nn.functional.interpolate(p5, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        fused = p4 + p5_up
        return self.fusion_conv(fused)

    def forward(self, features, class_idx):
        """
        Args:
            features: either a raw image tensor (B, 3, H, W) when use_cached_features=False,
                      or a dict {"c4": ..., "c5": ...} when use_cached_features=True.
            class_idx: (B,) integer tensor of class indices.

        Returns:
            bboxes: (B, num_queries, 4) — normalized corner [x, y, w, h] in [0, 1]
            plausibility: (B, num_queries) — logits (apply sigmoid for scores)
        """
        if not self.use_cached_features:
            with torch.no_grad():
                feat_dict = self.backbone(features)
            features = self._fuse_multiscale(feat_dict["c4"], feat_dict["c5"])
        elif isinstance(features, dict):
            features = self._fuse_multiscale(features["c4"], features["c5"])
        else:
            features = self.input_proj(features)

        B, C, H, W = features.shape
        pos = self.pos_encoder(features)
        features = features.flatten(2).permute(0, 2, 1)
        features = features + pos
        memory = self.transformer_encoder(features)

        class_emb = self.class_embed(class_idx)
        class_emb = class_emb.unsqueeze(1).expand(-1, self.num_queries, -1)
        queries = class_emb + self.query_offsets.unsqueeze(0)
        hs = self.transformer_decoder(tgt=queries, memory=memory)

        bboxes = self.bbox_head(hs)
        plausibility = self.plausibility_head(hs).squeeze(-1)
        return bboxes, plausibility
