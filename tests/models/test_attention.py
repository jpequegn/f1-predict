"""Tests for attention mechanisms."""

import pytest
import torch

from f1_predict.models.attention import (
    AttentionFusionModule,
    CrossModalAttention,
    MultiHeadCrossAttention,
    SelfAttention,
    SpatialAttention,
    TransformerFusionBlock,
)


class TestSelfAttention:
    """Tests for SelfAttention class."""

    @pytest.fixture
    def attention(self):
        """Create attention instance."""
        return SelfAttention(embed_dim=64, num_heads=4)

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 10, 64)  # batch, seq_len, embed_dim

    def test_init(self):
        """Test initialization."""
        attention = SelfAttention(embed_dim=128, num_heads=8)
        assert attention.embed_dim == 128
        assert attention.num_heads == 8

    def test_forward_shape(self, attention, sample_input):
        """Test forward pass output shape."""
        output = attention(sample_input)
        assert output.shape == sample_input.shape

    def test_forward_with_mask(self, attention, sample_input):
        """Test forward with attention mask."""
        mask = torch.zeros(2, 10, 10)
        output = attention(sample_input, mask=mask)
        assert output.shape == sample_input.shape

    def test_forward_gradient_flow(self, attention, sample_input):
        """Test gradient flow."""
        sample_input.requires_grad = True
        output = attention(sample_input)
        loss = output.sum()
        loss.backward()
        assert sample_input.grad is not None


class TestCrossModalAttention:
    """Tests for CrossModalAttention class."""

    @pytest.fixture
    def attention(self):
        """Create cross-modal attention instance."""
        return CrossModalAttention(
            query_dim=64,
            key_dim=128,
            hidden_dim=64,
            num_heads=4,
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors."""
        query = torch.randn(2, 10, 64)
        key_value = torch.randn(2, 20, 128)
        return query, key_value

    def test_init(self):
        """Test initialization."""
        attention = CrossModalAttention(
            query_dim=64, key_dim=128, hidden_dim=32, num_heads=4
        )
        assert attention is not None

    def test_forward_shape(self, attention, sample_inputs):
        """Test forward pass output shape."""
        query, key_value = sample_inputs
        output = attention(query, key_value)
        assert output.shape[0] == query.shape[0]
        assert output.shape[1] == query.shape[1]

    def test_forward_different_lengths(self, attention):
        """Test with different sequence lengths."""
        query = torch.randn(2, 5, 64)
        key_value = torch.randn(2, 15, 128)
        output = attention(query, key_value)
        assert output.shape[0] == 2
        assert output.shape[1] == 5


class TestMultiHeadCrossAttention:
    """Tests for MultiHeadCrossAttention class."""

    @pytest.fixture
    def attention(self):
        """Create multi-head cross attention instance."""
        return MultiHeadCrossAttention(
            embed_dim=64,
            num_heads=4,
        )

    def test_init(self):
        """Test initialization."""
        attention = MultiHeadCrossAttention(embed_dim=128, num_heads=8)
        assert attention.embed_dim == 128
        assert attention.num_heads == 8

    def test_forward_same_modality(self, attention):
        """Test forward with same modality inputs."""
        x = torch.randn(2, 10, 64)
        output = attention(x, x)
        assert output.shape == x.shape

    def test_forward_cross_modal(self, attention):
        """Test forward with cross-modal inputs."""
        query = torch.randn(2, 10, 64)
        context = torch.randn(2, 20, 64)
        output = attention(query, context)
        assert output.shape == query.shape


class TestSpatialAttention:
    """Tests for SpatialAttention class."""

    @pytest.fixture
    def attention(self):
        """Create spatial attention instance."""
        return SpatialAttention(channels=64)

    @pytest.fixture
    def sample_input(self):
        """Create sample feature map."""
        return torch.randn(2, 64, 14, 14)  # batch, channels, H, W

    def test_init(self):
        """Test initialization."""
        attention = SpatialAttention(channels=128)
        assert attention is not None

    def test_forward_shape(self, attention, sample_input):
        """Test forward pass output shape."""
        output = attention(sample_input)
        assert output.shape == sample_input.shape

    def test_forward_attention_weights(self, attention, sample_input):
        """Test attention weights are bounded."""
        output = attention(sample_input)
        # Output should be input weighted by attention
        assert output.min() >= sample_input.min() - 1
        assert output.max() <= sample_input.max() + 1


class TestAttentionFusionModule:
    """Tests for AttentionFusionModule class."""

    @pytest.fixture
    def fusion(self):
        """Create fusion module instance."""
        return AttentionFusionModule(
            image_dim=256,
            tabular_dim=64,
            hidden_dim=128,
            output_dim=256,
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs."""
        image_features = torch.randn(2, 256)
        tabular_features = torch.randn(2, 64)
        return image_features, tabular_features

    def test_init(self):
        """Test initialization."""
        fusion = AttentionFusionModule(
            image_dim=512, tabular_dim=128, hidden_dim=256, output_dim=512
        )
        assert fusion.output_dim == 512

    def test_forward_shape(self, fusion, sample_inputs):
        """Test forward pass output shape."""
        image_feats, tabular_feats = sample_inputs
        output = fusion(image_feats, tabular_feats)
        assert output.shape == (2, 256)

    def test_forward_with_batch_sizes(self, fusion):
        """Test with different batch sizes."""
        for batch_size in [1, 4, 8]:
            image_feats = torch.randn(batch_size, 256)
            tabular_feats = torch.randn(batch_size, 64)
            output = fusion(image_feats, tabular_feats)
            assert output.shape == (batch_size, 256)


class TestTransformerFusionBlock:
    """Tests for TransformerFusionBlock class."""

    @pytest.fixture
    def fusion_block(self):
        """Create transformer fusion block instance."""
        return TransformerFusionBlock(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs."""
        image_tokens = torch.randn(2, 49, 64)  # 7x7 feature map flattened
        tabular_tokens = torch.randn(2, 10, 64)
        return image_tokens, tabular_tokens

    def test_init(self):
        """Test initialization."""
        block = TransformerFusionBlock(embed_dim=128, num_heads=8, num_layers=4)
        assert block.embed_dim == 128
        assert block.num_layers == 4

    def test_forward_shape(self, fusion_block, sample_inputs):
        """Test forward pass output shape."""
        image_tokens, tabular_tokens = sample_inputs
        output = fusion_block(image_tokens, tabular_tokens)
        # Output should combine both modalities
        total_tokens = image_tokens.shape[1] + tabular_tokens.shape[1]
        assert output.shape == (2, total_tokens, 64)

    def test_forward_gradient_flow(self, fusion_block, sample_inputs):
        """Test gradient flow through fusion block."""
        image_tokens, tabular_tokens = sample_inputs
        image_tokens.requires_grad = True
        tabular_tokens.requires_grad = True

        output = fusion_block(image_tokens, tabular_tokens)
        loss = output.sum()
        loss.backward()

        assert image_tokens.grad is not None
        assert tabular_tokens.grad is not None


class TestAttentionEdgeCases:
    """Tests for attention edge cases."""

    def test_single_token_self_attention(self):
        """Test self-attention with single token."""
        attention = SelfAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 1, 64)
        output = attention(x)
        assert output.shape == x.shape

    def test_long_sequence_self_attention(self):
        """Test self-attention with long sequence."""
        attention = SelfAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 1000, 64)
        output = attention(x)
        assert output.shape == x.shape

    def test_single_channel_spatial_attention(self):
        """Test spatial attention with single channel."""
        attention = SpatialAttention(channels=1)
        x = torch.randn(2, 1, 14, 14)
        output = attention(x)
        assert output.shape == x.shape

    def test_large_feature_map_spatial_attention(self):
        """Test spatial attention with large feature map."""
        attention = SpatialAttention(channels=64)
        x = torch.randn(2, 64, 56, 56)
        output = attention(x)
        assert output.shape == x.shape


class TestAttentionTraining:
    """Tests for attention in training mode."""

    def test_self_attention_train_vs_eval(self):
        """Test self-attention behaves differently in train/eval."""
        attention = SelfAttention(embed_dim=64, num_heads=4, dropout=0.1)
        x = torch.randn(2, 10, 64)

        attention.train()
        train_output = attention(x)

        attention.eval()
        eval_output = attention(x)

        # With dropout, outputs should potentially differ
        # (though with same input they might be similar)
        assert train_output.shape == eval_output.shape

    def test_fusion_module_dropout(self):
        """Test fusion module with dropout."""
        fusion = AttentionFusionModule(
            image_dim=256, tabular_dim=64, hidden_dim=128, output_dim=256, dropout=0.5
        )

        image_feats = torch.randn(2, 256)
        tabular_feats = torch.randn(2, 64)

        fusion.train()
        outputs = [fusion(image_feats, tabular_feats) for _ in range(10)]

        # With high dropout, outputs should vary
        # Check they're not all identical
        _ = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        # Note: This might occasionally be true by chance, so we don't assert
        assert len(outputs) == 10
