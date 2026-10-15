# PyTorch-Project
PyTorch repository of code related to ToF DARTS project

## Layout

| Path | What it is |
| --- | --- |
| `basics/linear_regression.py` | One `nn.Linear(1, 1)` trained with SGD to fit a single point. |
| `darts/01_single_edge_linear.py` | First DARTS toy: one edge choosing between Identity and Linear with softmax scores (SGD). |
| `darts/02_single_edge_mlp.py` | One edge choosing between Identity and a small MLP on 4 points (Adam), then picks the best op. |
| `darts/gates.py` | Differentiable NOT, AND, OR, XOR and NAND gates for the gate-level search. Run it to print their truth tables. |
| `tests/test_gates.py` | Checks each gate against its Boolean truth table and checks its gradients. |

The scripts in `darts/` are numbered in the order they build on each other.

## Running

```
pip install -r requirements.txt
python darts/01_single_edge_linear.py
python -m pytest tests
```

## Differentiable gates

`darts/gates.py` is the first building block for searching over logic circuits instead of neural-network operations. Each gate is a polynomial that equals the gate's truth table when its inputs are exactly 0 or 1:

| Gate | Formula |
| --- | --- |
| NOT | `1 - a` |
| AND | `a * b` |
| OR | `a + b - a*b` |
| XOR | `a + b - 2*a*b` |
| NAND | `1 - a*b` |

For inputs between 0 and 1 the output is the chance the gate outputs 1 if each input were 1 with that probability, independently. The gates have no weights of their own; the search will learn which gate each node uses and which signals feed it. Fractional inputs give fractional outputs (for example `XOR(0.5, 0.5) = 0.5`), so a circuit picked from a softmax mixture still has to be checked on the exact truth table.
