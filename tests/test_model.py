import torch

from miniscale.config import ModelConfig
from model.transformer import Transformer, estimate_config_flops, estimate_model_flops


def tiny_config(checkpointing: bool = False) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=80,
        max_sequence_length=16,
        activation_checkpointing=checkpointing,
    )


def test_forward_is_causal_and_embeddings_are_tied() -> None:
    torch.manual_seed(1)
    model = Transformer(tiny_config()).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 20, 21]])
    with torch.no_grad():
        a = model(first).logits
        b = model(changed_future).logits
    torch.testing.assert_close(a[:, :2], b[:, :2])
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_shifted_language_model_loss_and_backward() -> None:
    model = Transformer(tiny_config(checkpointing=True)).train()
    tokens = torch.randint(0, 64, (2, 8))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 8, 64)
    assert output.loss is not None and output.loss.ndim == 0
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_flop_estimate_scales_with_tokens_and_parameters() -> None:
    model = Transformer(tiny_config())
    one = estimate_model_flops(model, tokens=10, sequence_length=8)
    two = estimate_model_flops(model, tokens=20, sequence_length=8)
    assert one > 0
    assert two == 2 * one
    assert estimate_config_flops(model.config, tokens=10, sequence_length=8) == one
