
import pytest
import torch

from lodd_sv.math.state_space import DiscreteStateSpace, VocabularyStateSpace
from lodd_sv.math.diffusion_equations import TransitionMatrixBuilder, get_qt_schedule
from lodd_sv.engine.forward_process import ForwardProcess
from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork
from lodd_sv.local_coding.local_loss import LocalPredictiveLoss, compute_local_losses
from lodd_sv.local_coding.async_optimizers import LayerWiseOptimizers
from lodd_sv.verification.uncertainty import entropy_per_position, uncertainty_score
from lodd_sv.verification.knowledge_base import InMemoryKnowledgeBase
from lodd_sv.verification.router import StatisticalRouter


def test_state_space():
    space = VocabularyStateSpace(vocab_size=100)
    assert space.state_dim == 100
    x = space.sample_uniform((3, 5), torch.device("cpu"))
    assert x.shape == (3, 5)
    oh = space.one_hot(x)
    assert oh.shape == (3, 5, 100)


def test_transition_matrix():
    builder = TransitionMatrixBuilder(10)
    beta = torch.tensor([0.1])
    Q = builder.uniform_transition(beta)
    assert Q.shape == (1, 10, 10)
    row_sum = Q[0].sum(dim=1)
    assert torch.allclose(row_sum, torch.ones(10))


def test_forward_process():
    space = VocabularyStateSpace(vocab_size=32)
    fp = ForwardProcess(space, num_steps=20)
    x_0 = torch.randint(0, 32, (2, 8))
    x_t, t = fp.corrupt(x_0)
    assert x_t.shape == x_0.shape
    assert t.shape == (2,)


def test_reverse_network():
    space = VocabularyStateSpace(vocab_size=32)
    model = ReverseDenoisingNetwork(space, d_model=16, num_layers=1, num_steps=20)
    x_t = torch.randint(0, 32, (2, 8))
    t = torch.randint(1, 21, (2,))
    logits, _ = model(x_t, t)
    assert logits.shape == (2, 8, 32)


def test_local_losses():
    space = VocabularyStateSpace(vocab_size=32)
    model = ReverseDenoisingNetwork(space, d_model=16, num_layers=2, num_steps=20)
    x_t = torch.randint(0, 32, (2, 8))
    t = torch.randint(1, 21, (2,))
    losses = compute_local_losses(model, x_t, t)
    assert len(losses) == 2
    assert all(lo.dim() == 0 for lo in losses)


def test_layer_optimizers_step():
    space = VocabularyStateSpace(vocab_size=32)
    model = ReverseDenoisingNetwork(space, d_model=16, num_layers=2, num_steps=20)
    opt = LayerWiseOptimizers(model, lr=1e-4)
    x_t = torch.randint(0, 32, (2, 8))
    t = torch.randint(1, 21, (2,))
    losses = compute_local_losses(model, x_t, t)
    metrics = opt.step(losses)
    assert "loss_l0" in metrics


def test_uncertainty():
    probs = torch.softmax(torch.randn(2, 4, 10), dim=-1)
    H = entropy_per_position(probs)
    assert H.shape == (2, 4)
    u = uncertainty_score(probs, method="entropy", aggregate="max")
    assert u.shape == (2,)


def test_knowledge_base():
    kb = InMemoryKnowledgeBase()
    kb.add("key1", "fact one")
    assert kb.get_fact("key1") == "fact one"
    results = kb.query("fact")
    assert len(results) >= 1


def test_router():
    kb = InMemoryKnowledgeBase()
    kb.add("a", "some fact")
    router = StatisticalRouter(kb, threshold=10.0)
    probs = torch.softmax(torch.randn(1, 5, 8), dim=-1)
    token_ids = torch.randint(0, 8, (1, 5))
    r = router(probs, token_ids)
    assert hasattr(r, "triggered")
    assert hasattr(r, "uncertainty_value")
