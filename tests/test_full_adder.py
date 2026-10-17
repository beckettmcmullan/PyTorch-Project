import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "darts"))
fa = importlib.import_module("04_full_adder")

# Textbook full adder over signals A=0, B=1, Cin=2, then nodes 3.. (n0..), plus one unused gate.
TEXTBOOK = [
    ("XOR", 0, 1),  # n0 = A xor B
    ("XOR", 3, 2),  # n1 = n0 xor Cin  -> Sum
    ("AND", 0, 1),  # n2 = A and B
    ("AND", 3, 2),  # n3 = n0 and Cin
    ("OR", 5, 6),  # n4 = n2 or n3     -> Cout
    ("NOT", 0, 0),  # n5 unused
]


def test_full_adder_table():
    x, y = fa.full_adder_table()
    assert (y[:, 0] + 2 * y[:, 1]).tolist() == x.sum(dim=1).tolist()


def test_run_discrete_textbook_adder():
    x, y = fa.full_adder_table()
    assert torch.equal(fa.run_discrete(TEXTBOOK, [1, 4], x), y)


def test_prune_drops_unused_gate():
    assert fa.prune(TEXTBOOK, [1, 4]) == [0, 1, 2, 3, 4]


def test_nodes_only_read_earlier_signals():
    model = fa.GateCircuits(num_circuits=4, num_nodes=5)
    for c in range(4):
        nodes, _ = model.extract(c)
        for i, (_, a, b) in enumerate(nodes):
            assert a < fa.NUM_INPUTS + i and b < fa.NUM_INPUTS + i


def test_one_hot_scores_reproduce_discrete_circuit():
    # With very confident scores the soft circuit must equal the extracted Boolean one.
    model = fa.GateCircuits(num_circuits=1, num_nodes=len(TEXTBOOK), init_std=0.0)
    with torch.no_grad():
        for i, (gate, a, b) in enumerate(TEXTBOOK):
            model.alpha[0, i, fa.GATE_NAMES.index(gate)] = 50
            model.beta_a[0, i, a] = 50
            model.beta_b[0, i, b] = 50
        model.gamma[0, 0, 1] = 50
        model.gamma[0, 1, 4] = 50
    x, y = fa.full_adder_table()
    assert model.extract(0) == (TEXTBOOK, [1, 4])
    assert torch.allclose(model(x)[0], y, atol=1e-6)


def test_short_search_runs():
    results = fa.search(circuits=2, num_nodes=5, steps=5)
    assert len(results) == 2
