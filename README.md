# PyTorch-Project
PyTorch repository of code related to ToF DARTS project

## Layout

| Path | What it is |
| --- | --- |
| `basics/linear_regression.py` | One `nn.Linear(1, 1)` trained with SGD to fit a single point. |
| `darts/01_single_edge_linear.py` | First DARTS toy: one edge choosing between Identity and Linear with softmax scores (SGD). |
| `darts/02_single_edge_mlp.py` | One edge choosing between Identity and a small MLP on 4 points (Adam), then picks the best op. |
| `darts/gates.py` | Differentiable NOT, AND, OR, XOR, NAND, NOR and XNOR gates for the gate-level search. Run it to print their truth tables. |
| `darts/04_full_adder.py` | Gate-level DARTS: searches for a full adder (A, B, Cin to Sum, Cout) out of the seven gates, then extracts and checks the discrete circuit. |
| `tests/test_gates.py` | Checks each gate against its Boolean truth table and checks its gradients. |
| `tests/test_full_adder.py` | Checks the full adder script's circuit evaluation, pruning and extraction. |

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
| NOR | `(1 - a)*(1 - b)` |
| XNOR | `1 - a - b + 2*a*b` |

For inputs between 0 and 1 the output is the chance the gate outputs 1 if each input were 1 with that probability, independently. The gates have no weights of their own; the search will learn which gate each node uses and which signals feed it. Fractional inputs give fractional outputs (for example `XOR(0.5, 0.5) = 0.5`), so a circuit picked from a softmax mixture still has to be checked on the exact truth table.

## Searching for a full adder (`darts/04_full_adder.py`)

```
cd darts
python 04_full_adder.py            # 64 searches at once, a few minutes on CPU
```

The circuit has the three inputs A, B, Cin and 8 gate nodes. Each node has:

- **alpha**: scores over the 7 gates (NOT, AND, OR, XOR, NAND, NOR, XNOR),
- **beta_a, beta_b**: scores over which signal feeds each of its two inputs. A node can read the circuit inputs and earlier nodes only, so the circuit is always acyclic. NOT ignores its second input.

Two **gamma** score vectors pick which node drives Sum and which drives Cout. Each choice is a softmax mixture during training, so the whole circuit is differentiable and is trained on the 8-row truth table with binary cross-entropy. Afterwards every choice is snapped to its argmax, gates the outputs do not depend on are pruned, and the discrete circuit is checked on the exact truth table. Only circuits that pass that check count as found.

What it took to get exact circuits out (share of 64 seeds whose snapped circuit is an exact adder, 3000 Adam steps, lr 0.05):

| Setting | Exact |
| --- | --- |
| 6 gates, scores start near 0, temperature 1 to 0.05 | 0% |
| 8 gates, random scores (std 1) | 8% |
| + Gumbel noise on the scores during training | 20% |
| + temperature only annealed down to 0.3 (the default) | 23% |
| 10 gates, with noise | 25% |
| 12 gates, no noise | 27% |

Without these, nearly every search gets stuck with Sum right and Cout copying one input, which matches 6 of the 8 rows. Gumbel noise keeps the search exploring, and the temperature anneal makes the end-of-training mixture close to the circuit that gets extracted.

The default run finds an exact adder in 15 of 64 searches. The smallest is 5 gates, the minimum for a full adder with two-input gates:

```
n0 = XNOR(Cin, B)
n1 = OR(n0, A)
n2 = OR(Cin, B)
n5 = XNOR(n0, A)        Sum  = A xor B xor Cin
n6 = AND(n2, n1)        Cout = (B or Cin) and (A or B == Cin)
```

This is not the textbook circuit (two XORs, two ANDs, one OR) but it is equivalent: when B equals Cin, Cout is B; otherwise Cout is A. Most other hits use 6 or 7 gates, often with a redundant gate (for example `OR(Cin, Cin)`), since nothing in the loss rewards a smaller circuit yet. A hardware cost term would.
