import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "darts"))
from gates import GATES, apply_gate, gate_and, gate_nand, gate_not, gate_or, gate_xor, truth_table  # noqa: E402

EXPECTED = {
    "NOT": [1, 0],
    "AND": [0, 0, 0, 1],
    "OR": [0, 1, 1, 1],
    "XOR": [0, 1, 1, 0],
    "NAND": [1, 1, 1, 0],
}


def test_truth_table_rows():
    assert truth_table(2).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


@pytest.mark.parametrize("name", GATES)
def test_matches_boolean_truth_table(name):
    _, arity = GATES[name]
    out = apply_gate(name, truth_table(arity))
    assert out.tolist() == EXPECTED[name]


@pytest.mark.parametrize("name", GATES)
def test_output_stays_in_unit_interval(name):
    _, arity = GATES[name]
    inputs = torch.rand(1000, arity)
    out = apply_gate(name, inputs)
    assert out.min() >= 0 and out.max() <= 1


# Hand-derived partial derivatives (d/da, d/db) for each two-input gate.
PARTIALS = {
    gate_and: lambda a, b: (b, a),
    gate_or: lambda a, b: (1 - b, 1 - a),
    gate_xor: lambda a, b: (1 - 2 * b, 1 - 2 * a),
    gate_nand: lambda a, b: (-b, -a),
}


@pytest.mark.parametrize("fn", PARTIALS, ids=lambda f: f.__name__)
def test_gradients_match_hand_derived(fn):
    a = torch.rand(50, requires_grad=True)
    b = torch.rand(50, requires_grad=True)
    fn(a, b).sum().backward()
    da, db = PARTIALS[fn](a.detach(), b.detach())
    assert torch.allclose(a.grad, da)
    assert torch.allclose(b.grad, db)


def test_not_gradient():
    a = torch.rand(50, requires_grad=True)
    gate_not(a).sum().backward()
    assert torch.allclose(a.grad, -torch.ones(50))


@pytest.mark.parametrize("name", GATES)
def test_gradcheck(name):
    fn, arity = GATES[name]
    inputs = [torch.rand(10, dtype=torch.float64, requires_grad=True) for _ in range(arity)]
    assert torch.autograd.gradcheck(fn, inputs)
