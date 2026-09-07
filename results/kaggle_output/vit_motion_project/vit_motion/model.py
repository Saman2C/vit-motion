from __future__ import annotations

import timm
import torch
from torch import nn


class ViTMotionModel(nn.Module):
    """One cached RGB token + image age + K numeric tokens -> current delta."""

    def __init__(
        self,
        encoder_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        sequence_length: int = 6,
        image_update_interval: int = 1,
        temporal_d_model: int = 64,
        temporal_nhead: int = 4,
        temporal_num_layers: int = 2,
        temporal_dim_feedforward: int = 128,
        temporal_dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if temporal_d_model % temporal_nhead:
            raise ValueError("temporal_d_model must be divisible by temporal_nhead")
        self.sequence_length = int(sequence_length)
        if image_update_interval < 1:
            raise ValueError("image_update_interval must be at least 1")
        self.image_update_interval = int(image_update_interval)
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(self.encoder.num_features, temporal_d_model),
            nn.LayerNorm(temporal_d_model),
            nn.GELU(),
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(5, temporal_d_model),
            nn.LayerNorm(temporal_d_model),
            nn.GELU(),
        )
        self.image_age_projection = nn.Sequential(
            nn.Linear(1, temporal_d_model),
            nn.GELU(),
            nn.Linear(temporal_d_model, temporal_d_model),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.sequence_length + 1, temporal_d_model)
        )
        self.token_type_embedding = nn.Embedding(2, temporal_d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_d_model,
            nhead=temporal_nhead,
            dim_feedforward=temporal_dim_feedforward,
            dropout=temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_num_layers,
            norm=nn.LayerNorm(temporal_d_model),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(temporal_d_model, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(temporal_dropout),
            nn.Linear(head_hidden_dim, 3),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual_projection(self.encoder(image))

    def predict_from_feature(
        self,
        visual_feature: torch.Tensor,
        numeric_sequence: torch.Tensor,
        image_age: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if numeric_sequence.ndim != 3:
            raise ValueError("numeric_sequence must have shape [B, K, 5]")
        if numeric_sequence.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected K={self.sequence_length}, got K={numeric_sequence.shape[1]}"
            )
        if image_age is None:
            image_age = torch.zeros(
                visual_feature.shape[0], device=visual_feature.device, dtype=visual_feature.dtype
            )
        image_age = image_age.reshape(-1, 1).to(
            device=visual_feature.device, dtype=visual_feature.dtype
        )
        if image_age.shape[0] != visual_feature.shape[0]:
            raise ValueError("image_age must contain one value per batch item")
        visual_token = (
            visual_feature + self.image_age_projection(image_age)
        ).unsqueeze(1)
        numeric_tokens = self.numeric_projection(numeric_sequence)
        tokens = torch.cat((visual_token, numeric_tokens), dim=1)
        type_ids = torch.cat(
            (
                torch.zeros(1, device=tokens.device, dtype=torch.long),
                torch.ones(self.sequence_length, device=tokens.device, dtype=torch.long),
            )
        )
        tokens = tokens + self.position_embedding + self.token_type_embedding(type_ids)[None]
        encoded = self.temporal_encoder(tokens)
        return self.motion_head(encoded[:, -1, :])

    def forward(
        self,
        image: torch.Tensor,
        numeric_sequence: torch.Tensor,
        image_age: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_from_feature(
            self.encode_image(image), numeric_sequence, image_age
        )
