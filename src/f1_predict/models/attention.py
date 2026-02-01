"""Attention mechanisms for multi-modal F1 prediction.

Implements self-attention, cross-modal attention, and multi-head attention
for better feature fusion and interpretability.
"""

import logging
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    """Self-attention layer for feature refinement.

    Allows features to attend to other features within the same modality,
    learning important relationships and dependencies.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Initialize self-attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in projections
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # For storing attention weights (visualization)
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Store for visualization
        self.attention_weights = attn_weights.detach()

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = self.out_proj(out)

        if return_attention:
            return out, attn_weights
        return out, None


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing image and tabular features.

    Allows one modality to attend to the other, learning what information
    from one modality is most relevant given the other.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
    ):
        """Initialize cross-modal attention.

        Args:
            query_dim: Dimension of query modality
            key_dim: Dimension of key/value modality
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_dim: Output dimension (defaults to query_dim)
        """
        super().__init__()

        self.output_dim = output_dim or query_dim
        self.num_heads = num_heads

        # Compute head dimension based on output_dim
        assert self.output_dim % num_heads == 0
        self.head_dim = self.output_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(query_dim, self.output_dim)
        self.k_proj = nn.Linear(key_dim, self.output_dim)
        self.v_proj = nn.Linear(key_dim, self.output_dim)
        self.out_proj = nn.Linear(self.output_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)

        # Store attention weights
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            query: Query tensor from first modality (batch_size, query_len, query_dim)
            key_value: Key/Value tensor from second modality (batch_size, kv_len, key_dim)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (attended features, optional attention weights)
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        kv_len = key_value.shape[1]

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        # Reshape for multi-head
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        self.attention_weights = attn_weights.detach()

        # Apply to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.output_dim)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_weights
        return out, None


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention layer.

    Can be used for both self-attention and cross-attention depending
    on how inputs are provided.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.scale = self.head_dim ** -0.5

        # Combined projection for efficiency
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            query: Query tensor (batch_size, query_len, embed_dim)
            key: Key tensor (batch_size, key_len, embed_dim)
            value: Value tensor (batch_size, value_len, embed_dim)
            key_padding_mask: Mask for padded keys
            need_weights: Whether to return attention weights
            attn_mask: Additional attention mask

        Returns:
            Tuple of (output, optional attention weights)
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]

        # Project Q, K, V
        if self.in_proj_bias is not None:
            qkv = F.linear(
                torch.cat([query, key, value], dim=1),
                self.in_proj_weight,
                self.in_proj_bias,
            )
        else:
            qkv = F.linear(
                torch.cat([query, key, value], dim=1),
                self.in_proj_weight,
            )

        # Split Q, K, V
        q, k, v = qkv.split([query_len, key_len, key_len], dim=1)
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        out = self.out_proj(out)

        if need_weights:
            return out, attn_weights.mean(dim=1)  # Average over heads
        return out, None


class SpatialAttention(nn.Module):
    """Spatial attention for image features.

    Learns to focus on important spatial regions of an image,
    useful for highlighting relevant parts of telemetry plots.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """Initialize spatial attention.

        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction ratio
        """
        super().__init__()

        # Channel attention (squeeze-excitation style)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.

        Args:
            x: Input feature map (batch_size, C, H, W)

        Returns:
            Attended feature map
        """
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights.unsqueeze(-1).unsqueeze(-1)

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_features)
        x = x * spatial_weights

        return x


class AttentionFusionModule(nn.Module):
    """Attention-based fusion of image and tabular features.

    Uses cross-modal attention to allow each modality to selectively
    attend to relevant information from the other.
    """

    def __init__(
        self,
        image_dim: int,
        tabular_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        output_dim: int = 256,
    ):
        """Initialize attention fusion module.

        Args:
            image_dim: Dimension of image features
            tabular_dim: Dimension of tabular features
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super().__init__()

        self.output_dim = output_dim

        # Project both modalities to same dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)

        # Cross-attention: image attends to tabular
        self.image_to_tabular = CrossModalAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_dim=hidden_dim,
        )

        # Cross-attention: tabular attends to image
        self.tabular_to_image = CrossModalAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_dim=hidden_dim,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

        # Store attention weights for visualization
        self.img_to_tab_weights: Optional[torch.Tensor] = None
        self.tab_to_img_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        image_features: torch.Tensor,
        tabular_features: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Fuse image and tabular features with attention.

        Args:
            image_features: Image features (batch_size, image_dim) or (batch_size, seq, image_dim)
            tabular_features: Tabular features (batch_size, tabular_dim)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (fused features, optional attention dict)
        """
        # Ensure 3D for attention
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        if tabular_features.dim() == 2:
            tabular_features = tabular_features.unsqueeze(1)

        # Project
        img_proj = self.image_proj(image_features)
        tab_proj = self.tabular_proj(tabular_features)

        # Cross-attention
        img_attended, img_attn = self.image_to_tabular(
            img_proj, tab_proj, return_attention=True
        )
        tab_attended, tab_attn = self.tabular_to_image(
            tab_proj, img_proj, return_attention=True
        )

        # Store attention
        self.img_to_tab_weights = img_attn
        self.tab_to_img_weights = tab_attn

        # Concatenate all representations
        # Original + attended
        combined = torch.cat([
            img_proj.squeeze(1),
            tab_proj.squeeze(1),
            img_attended.squeeze(1),
            tab_attended.squeeze(1),
        ], dim=-1)

        # Fuse
        fused = self.fusion(combined)

        if return_attention:
            attn_dict = {
                "image_to_tabular": img_attn,
                "tabular_to_image": tab_attn,
            }
            return fused, attn_dict

        return fused, None


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data.

    Adds positional information to embeddings, useful for
    temporal sequences of telemetry data.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerFusionBlock(nn.Module):
    """Transformer-style fusion block for multi-modal learning.

    Combines self-attention and cross-attention with feedforward
    networks for powerful feature fusion.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """Initialize transformer fusion block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Self-attention
        self.self_attn = SelfAttention(d_model, num_heads, dropout)

        # Cross-modal attention
        self.cross_attn = CrossModalAttention(d_model, d_model, num_heads, dropout)

        # Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query features (batch_size, seq_len, d_model)
            context: Context features from other modality

        Returns:
            Fused features
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention with residual
        cross_out, _ = self.cross_attn(x, context)
        x = self.norm2(x + self.dropout(cross_out))

        # Feedforward with residual
        ff_out = self.feedforward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x
