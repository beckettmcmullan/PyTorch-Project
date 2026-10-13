# PyTorch-Project
PyTorch repository of code related to ToF DARTS project

## Layout

| Path | What it is |
| --- | --- |
| `basics/linear_regression.py` | One `nn.Linear(1, 1)` trained with SGD to fit a single point. |
| `darts/01_single_edge_linear.py` | First DARTS toy: one edge choosing between Identity and Linear with softmax scores (SGD). |
| `darts/02_single_edge_mlp.py` | One edge choosing between Identity and a small MLP on 4 points (Adam), then picks the best op. |

The scripts in `darts/` are numbered in the order they build on each other.

## Running

```
pip install -r requirements.txt
python darts/01_single_edge_linear.py
```
