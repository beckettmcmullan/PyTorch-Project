"""Differentiable versions of primitive Boolean logic gates.

Each gate works on signals in [0, 1], where 0 means False and 1 means True.
On exact 0/1 inputs the output matches the gate's truth table. On fractional
inputs (which is what a softmax mixture produces during search) the output is
the probability the gate outputs 1 if its inputs were independent coin flips
with those probabilities. Every formula is a polynomial, so gradients exist
everywhere.

The gates have no learnable weights. In the gate-level search, the learnable
parts are the scores that pick which gate a node uses (alpha) and which
signals feed its inputs (beta), not the gates themselves.
"""

import torch


def gate_not(a):
    return 1 - a


def gate_and(a, b):
    return a * b


def gate_or(a, b):
    return a + b - a * b


def gate_xor(a, b):
    return a + b - 2 * a * b


def gate_nand(a, b):
    return 1 - a * b


# name -> (function, number of inputs). NOT is the only unary gate.
GATES = {
    "NOT": (gate_not, 1),
    "AND": (gate_and, 2),
    "OR": (gate_or, 2),
    "XOR": (gate_xor, 2),
    "NAND": (gate_nand, 2),
}


def truth_table(num_inputs):
    """Every 0/1 input combination as a float tensor of shape (2**n, n)."""
    rows = [[(i >> bit) & 1 for bit in reversed(range(num_inputs))] for i in range(2 ** num_inputs)]
    return torch.tensor(rows, dtype=torch.float32)


def apply_gate(name, inputs):
    """Apply a gate to the columns of `inputs` (shape (rows, num_inputs))."""
    fn, arity = GATES[name]
    return fn(*[inputs[:, i] for i in range(arity)])


if __name__ == "__main__":
    for name, (_, arity) in GATES.items():
        table = truth_table(arity)
        out = apply_gate(name, table)
        print(name)
        for row, y in zip(table.int().tolist(), out.int().tolist()):
            print(f"  {row} -> {y}")
    # Fractional inputs give fractional outputs: this is the relaxation gap.
    a, b = torch.tensor(0.5), torch.tensor(0.5)
    print(f"XOR(0.5, 0.5) = {gate_xor(a, b).item()}  (not a valid Boolean value)")
