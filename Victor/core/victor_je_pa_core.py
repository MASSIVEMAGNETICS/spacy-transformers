"""
victor_je_pa_core.py
====================

This module implements a lightweight Joint-Embedding Predictive Architecture (JEPA)
for learning a world model from screen recordings.  It follows the high-level
design of I-JEPA (Image-based JEPA) introduced by** Caron et al.**, where a vision
transformer encodes a partially observed context block and a predictor network
produces latent representations of target blocks【122389235253242†L140-L170】.  The target
encoder weights are an exponential moving average (EMA) of the context encoder
to prevent representation collapse【122389235253242†L140-L171】.  The model predicts
representations in latent space instead of pixels, making it more efficient
than pixel-reconstruction methods【122389235253242†L214-L222】.  A pretrained
DINOv2 backbone is recommended for the context/target encoders because
these models provide high-performance visual features that can be used with
simple linear classifiers without fine-tuning【628919324280872†L66-L99】.

The implementation below is a simplified scaffold intended to get you
started.  It uses PyTorch, torchvision and basic utilities to build a JEPA
model.  To train on your own screen recordings you will need to prepare a
dataset of video frames (e.g. extracted as PNG images).  The script defines
data loaders, the JEPA model, EMA update logic and a basic training loop.

Key components
--------------

1. **ScreenRecordingDataset** – reads images from a directory and applies
   transformations.
2. **JEPA** – encapsulates context and target encoders (both Vision
   Transformers) and a predictor network.  The context encoder is trained
   with gradients, while the target encoder is updated via EMA to prevent
   collapse【122389235253242†L150-L170】.
3. **Mask sampling** – randomly selects multiple target blocks (e.g. four
   per image) and a larger context block, removing overlapping patches from
   the context【122389235253242†L186-L238】.
4. **Training loop** – computes loss between predicted and actual latent
   representations using mean squared error【122389235253242†L214-L222】.

This code is designed to be run on a machine with GPU support and
installed dependencies (PyTorch >= 2.0, torchvision).  It does not
automatically download DINOv2 weights; you can replace the encoders with
`torchvision.models.vit_b_16` or load your own pretrained backbone using
`timm` or the DINOv2 repository.

Note
----

This is a research prototype, not a production-ready implementation.  It
omits some optimizations (e.g. multi-block sampling strategies,
distributed training) and is intentionally small (<50M parameters).  Use
this as a starting point to build a more sophisticated JEPA tailored to
your screen recordings.
"""

from __future__ import annotations

import os
import math
import random
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class ScreenRecordingDataset(Dataset):
    """Dataset for loading screen recording frames from a directory.

    Each item returns a PIL image and its corresponding tensor.  You can
    extend this class to include timestamps, mouse/keyboard actions, etc.
    """

    def __init__(self, root: str, image_size: int = 224) -> None:
        super().__init__()
        self.root = root
        self.paths: List[str] = []
        for fname in os.listdir(root):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.paths.append(os.path.join(root, fname))
        self.paths.sort()
        # Basic transformations: resize and normalize to [-1,1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def get_vit_backbone(name: str = "vit_b_16", pretrained: bool = True) -> nn.Module:
    """Load a Vision Transformer backbone.

    Args:
        name: model name.  Use "vit_b_16" (ViT-B/16) or other available
            torchvision models.  Set pretrained=True to load default weights
            trained on ImageNet-1K.  Replace with DINOv2 for better features.

    Returns:
        A model that outputs patch-level features (sequence of tokens).
    """
    if name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "vit_l_16":
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported ViT backbone: {name}")
    return model


def ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    """Update target parameters towards source using exponential moving average.

    The target encoder receives no gradients; its weights are an EMA of the
    context encoder weights to avoid trivial solutions【122389235253242†L150-L171】.
    """
    with torch.no_grad():
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.mul_(decay).add_(src_param.data * (1 - decay))


class PositionEmbedding(nn.Module):
    """Simple positional embedding for target block coordinates.

    Encodes the (x, y, width, height) of a block into a fixed-dimensional vector.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(4, embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, 4) normalized [0,1] values for (x,y,w,h)
        return self.linear(coords)


class Predictor(nn.Module):
    """Predictor network that maps context representations and positional tokens
    to predicted target representations.

    A lightweight MLP is used here, but you could replace this with a
    transformer for better capacity【122389235253242†L158-L165】.
    """

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + embed_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, context_repr: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        # context_repr: (B, D), pos_embed: (B, D)
        x = torch.cat([context_repr, pos_embed], dim=-1)
        return self.mlp(x)


class JEPA(nn.Module):
    """Joint-Embedding Predictive Architecture model.

    It contains a context encoder, an EMA target encoder, a position embedding
    module and a predictor network.  For simplicity the encoders return a

    single CLS token representation; you can adapt this to patch-level tokens.
    """

    def __init__(self, backbone_name: str = "vit_b_16", embed_dim: int = 768, ema_decay: float = 0.996) -> None:
        super().__init__()
        # Context encoder (learnable)
        self.context_encoder = get_vit_backbone(backbone_name, pretrained=True)
        # Target encoder (EMA, not trained directly)
        self.target_encoder = get_vit_backbone(backbone_name, pretrained=True)
        # Freeze target encoder gradients
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_decay = ema_decay
        # Position embedding module for 4-element bounding boxes
        self.pos_emb = PositionEmbedding(embed_dim)
        # Predictor network
        self.predictor = Predictor(embed_dim, embed_dim)

    def forward(self, images: torch.Tensor, targets: List[Tuple[float, float, float, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through JEPA.

        Args:
            images: (B, C, H, W) batch of images.
            targets: list of bounding boxes for each image.  Each box is
                normalized (x, y, w, h) where x and y denote the top-left
                coordinate as fractions of image dimensions.  This simplified
                interface assumes one target per image.  Extending to multiple
                targets is straightforward.

        Returns:
            mse_loss: mean squared error between predicted and actual target
                representations.
            predictions: predicted target representations (for logging).
        """
        B = images.size(0)
        device = images.device
        # Context encoder processes masked images (context); here we simply
        # zero-out pixels inside target bounding boxes.  For multiple targets
        # you should mask all targets.
        masked_images = images.clone()
        for i, (x, y, w, h) in enumerate(targets):
            # Convert relative coords to pixel indices
            H, W = images.shape[2], images.shape[3]
            x0 = int(x * W)
            y0 = int(y * H)
            x1 = int((x + w) * W)
            y1 = int((y + h) * H)
            masked_images[i, :, y0:y1, x0:x1] = 0.0
        # Obtain context representation (CLS token at index 0)
        ctx_out = self.context_encoder(masked_images)
        if isinstance(ctx_out, tuple):  # torchvision returns (logits)
            ctx_repr = ctx_out[0]
        else:
            ctx_repr = ctx_out
        # Flatten to (B, D)
        ctx_repr = ctx_repr.squeeze()
        # Target representation using EMA encoder (CLS token)
        tgt_out = self.target_encoder(images)
        if isinstance(tgt_out, tuple):
            tgt_repr = tgt_out[0]
        else:
            tgt_repr = tgt_out
        tgt_repr = tgt_repr.squeeze()
        # Positional embeddings for each target
        coords = torch.tensor(targets, device=device)
        pos_embed = self.pos_emb(coords)
        # Predictions
        pred = self.predictor(ctx_repr, pos_embed)
        # Mean squared error loss
        loss = F.mse_loss(pred, tgt_repr)
        return loss, pred

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Update the target encoder weights using EMA of the context encoder."""
        ema_update(self.target_encoder, self.context_encoder, self.ema_decay)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into their latent representations using the target encoder."""
        tgt_out = self.target_encoder(images)
        if isinstance(tgt_out, tuple):
            tgt_repr = tgt_out[0]
        else:
            tgt_repr = tgt_out
        return tgt_repr.squeeze()


def sample_random_boxes(batch_size: int, min_scale: float = 0.1, max_scale: float = 0.3) -> List[Tuple[float, float, float, float]]:
    """Sample random bounding boxes for JEPA target blocks.

    Boxes are defined by (x,y,w,h) in normalized [0,1] coordinates.  The sizes
    are randomly chosen between `min_scale` and `max_scale` of the image area.
    """
    boxes: List[Tuple[float, float, float, float]] = []
    for _ in range(batch_size):
        w = random.uniform(min_scale, max_scale)
        h = random.uniform(min_scale, max_scale)
        x = random.uniform(0.0, 1.0 - w)
        y = random.uniform(0.0, 1.0 - h)
        boxes.append((x, y, w, h))
    return boxes


def train_jepa(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    backbone_name: str = "vit_b_16",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> JEPA:
    """Train the JEPA model on screen recordings.

    Args:
        data_dir: path to directory containing frame images.
        epochs: number of training epochs.
        batch_size: mini-batch size.
        learning_rate: optimizer learning rate.
        backbone_name: ViT backbone identifier.
        device: computation device.

    Returns:
        Trained JEPA model.
    """
    dataset = ScreenRecordingDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # Determine embedding dimension from backbone; vit_b_16 has embed_dim=768
    embed_dim = 768 if backbone_name == "vit_b_16" else 1024
    model = JEPA(backbone_name=backbone_name, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.context_encoder.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images in loader:
            images = images.to(device)
            # Sample random target boxes (one per image)
            boxes = sample_random_boxes(images.size(0))
            optimizer.zero_grad()
            loss, _ = model(images, boxes)
            loss.backward()
            optimizer.step()
            # Update EMA target encoder
            model.update_target_encoder()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return model


if __name__ == "__main__":
    # Example usage: train on frames stored in ./data/screen_recordings
    import argparse
    parser = argparse.ArgumentParser(description="Train a JEPA world model on screen recordings.")
    parser.add_argument("data_dir", type=str, help="Directory containing frame images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="vit_b_16")
    args = parser.parse_args()
    train_jepa(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, backbone_name=args.backbone)
