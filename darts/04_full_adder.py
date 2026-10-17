"""Gate-level DARTS: search for a full adder built from differentiable logic gates.

The circuit has three inputs (A, B, Cin) followed by a fixed number of gate
nodes. Every gate node has three sets of learnable scores:

  alpha    which gate the node is (softmax over NOT, AND, OR, XOR, NAND, NOR, XNOR)
  beta_a   which earlier signal feeds the node's first input
  beta_b   which earlier signal feeds the node's second input

A node can only read from the circuit inputs and from nodes before it, so the
circuit is always acyclic. Two more score vectors (gamma) pick which node drives
each output, Sum and Cout.

During search every choice is a softmax mixture, so the whole circuit is a
differentiable function of the scores and is trained with gradient descent on
the 8-row truth table. After search each choice is snapped to its argmax,
unused gates are pruned, and the discrete circuit is checked on the exact truth
table. The soft loss alone proves nothing: a mixture can fit the table while
its argmax circuit does not.

What helped, measured as the share of 64 seeds whose snapped circuit is an
exact adder (3000 Adam steps each, see README for the full table):
  * Random initial scores (std 1) and spare gates. With near-zero scores and
    6 gates, 0 of 64 worked: seeds typically settle on Sum right and Cout
    copying one input (e.g. Cout = Cin), which is right on 6 of 8 rows. Random scores with 8 gates: 8%.
  * Gumbel noise on the scores during training (Gumbel-softmax): 20%.
  * Annealing the softmax temperature from 1 to 0.3, so the mixtures end
    close to one-hot and the trained circuit is close to the extracted one:
    23%. More gates help too (12 gates without noise: 27%).

Even then only some seeds find an exact adder, so the script searches many
seeds at once (one batch of independent circuits) and reports every one whose
discrete circuit passes the exact check.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from gates import GATES, truth_table

GATE_NAMES = list(GATES)
INPUT_NAMES = ["A", "B", "Cin"]
OUTPUT_NAMES = ["Sum", "Cout"]
NUM_INPUTS = len(INPUT_NAMES)


def full_adder_table():
    x = truth_table(3)  # columns A, B, Cin
    total = x.sum(dim=1)
    y = torch.stack([total % 2, (total >= 2).float()], dim=1)  # Sum, Cout
    return x, y


def gumbel(t):
    return -torch.log(-torch.log(torch.rand_like(t).clamp(1e-9, 1 - 1e-9)))


class GateCircuits(nn.Module):
    """`num_circuits` independent searchable circuits, trained together as one batch."""

    def __init__(self, num_circuits, num_nodes=8, num_outputs=2, init_std=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        sources = NUM_INPUTS + num_nodes
        self.alpha = nn.Parameter(init_std * torch.randn(num_circuits, num_nodes, len(GATE_NAMES)))
        self.beta_a = nn.Parameter(init_std * torch.randn(num_circuits, num_nodes, sources))
        self.beta_b = nn.Parameter(init_std * torch.randn(num_circuits, num_nodes, sources))
        self.gamma = nn.Parameter(init_std * torch.randn(num_circuits, num_outputs, num_nodes))
        # Node i may read the circuit inputs and nodes 0..i-1 only (keeps the circuit acyclic).
        mask = torch.full((num_nodes, sources), float("-inf"))
        for i in range(num_nodes):
            mask[i, : NUM_INPUTS + i] = 0
        self.register_buffer("mask", mask)

    def forward(self, x, tau=1.0, noise=0.0):
        def weights(scores, extra=0):
            if noise:
                scores = scores + noise * gumbel(scores)
            return F.softmax((scores + extra) / tau, dim=-1)

        w_gate = weights(self.alpha)
        w_a = weights(self.beta_a, self.mask)
        w_b = weights(self.beta_b, self.mask)
        w_out = weights(self.gamma)

        circuits = self.alpha.shape[0]
        signals = x.new_zeros(circuits, x.shape[0], NUM_INPUTS + self.num_nodes)  # (circuits, rows, sources)
        signals[:, :, :NUM_INPUTS] = x
        for i in range(self.num_nodes):
            a = torch.einsum("crs,cs->cr", signals, w_a[:, i])
            b = torch.einsum("crs,cs->cr", signals, w_b[:, i])
            outs = torch.stack([fn(a) if arity == 1 else fn(a, b) for fn, arity in GATES.values()], dim=-1)
            signals = signals.clone()  # autograd needs the earlier version intact
            signals[:, :, NUM_INPUTS + i] = (outs * w_gate[:, i, None, :]).sum(-1)
        return torch.einsum("crn,con->cro", signals[:, :, NUM_INPUTS:], w_out)  # (circuits, rows, outputs)

    def extract(self, c):
        """Argmax of every choice for circuit c: a list of (gate, src_a, src_b) and the output nodes."""
        gates = self.alpha[c].argmax(-1).tolist()
        src_a = (self.beta_a[c] + self.mask).argmax(-1).tolist()
        src_b = (self.beta_b[c] + self.mask).argmax(-1).tolist()
        nodes = [(GATE_NAMES[g], a, b) for g, a, b in zip(gates, src_a, src_b)]
        return nodes, self.gamma[c].argmax(-1).tolist()


def run_discrete(nodes, outputs, x):
    """Evaluate an extracted circuit with plain Boolean gates (no mixtures)."""
    signals = [x[:, i] for i in range(NUM_INPUTS)]
    for gate, a, b in nodes:
        fn, arity = GATES[gate]
        signals.append(fn(signals[a]) if arity == 1 else fn(signals[a], signals[b]))
    return torch.stack([signals[NUM_INPUTS + o] for o in outputs], dim=1)


def prune(nodes, outputs):
    """Indices of the gate nodes the outputs actually depend on."""
    used, stack = set(), list(outputs)
    while stack:
        i = stack.pop()
        if i in used:
            continue
        used.add(i)
        gate, a, b = nodes[i]
        for src in [a] if GATES[gate][1] == 1 else [a, b]:
            if src >= NUM_INPUTS:
                stack.append(src - NUM_INPUTS)
    return sorted(used)


def describe(nodes, outputs):
    def name(src):
        return INPUT_NAMES[src] if src < NUM_INPUTS else f"n{src - NUM_INPUTS}"

    lines = []
    for i in prune(nodes, outputs):
        gate, a, b = nodes[i]
        args = name(a) if GATES[gate][1] == 1 else f"{name(a)}, {name(b)}"
        lines.append(f"n{i} = {gate}({args})")
    for out_name, o in zip(OUTPUT_NAMES, outputs):
        lines.append(f"{out_name} = n{o}")
    return lines


def search(seed=0, circuits=64, num_nodes=8, steps=3000, lr=0.05, tau_start=1.0, tau_end=0.3, noise=1.0):
    torch.manual_seed(seed)
    x, y = full_adder_table()
    model = GateCircuits(circuits, num_nodes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        tau = tau_start * (tau_end / tau_start) ** (step / (steps - 1))  # geometric anneal
        pred = model(x, tau, noise).clamp(1e-6, 1 - 1e-6)
        # Sum over circuits so each circuit gets the same gradient it would get alone.
        loss = F.binary_cross_entropy(pred, y.expand_as(pred), reduction="none").mean(dim=(1, 2)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    results = []
    with torch.no_grad():
        for c in range(circuits):
            nodes, outputs = model.extract(c)
            exact = torch.equal(run_discrete(nodes, outputs, x), y)
            results.append((nodes, outputs, exact))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--circuits", type=int, default=64, help="independent searches run as one batch")
    parser.add_argument("--nodes", type=int, default=8, help="gate nodes per circuit")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=1.0, help="Gumbel noise scale (0 = plain softmax)")
    args = parser.parse_args()

    results = search(args.seed, args.circuits, args.nodes, args.steps, noise=args.noise)
    exact = [(nodes, outputs) for nodes, outputs, ok in results if ok]
    print(f"{len(exact)}/{len(results)} searches extracted an exact full adder\n")

    # Show each distinct exact circuit once, smallest first.
    seen = {}
    for nodes, outputs in exact:
        lines = tuple(describe(nodes, outputs))
        seen.setdefault(lines, 0)
        seen[lines] += 1
    for lines, count in sorted(seen.items(), key=lambda kv: len(kv[0])):
        print(f"{len(lines) - len(OUTPUT_NAMES)} gates (found {count}x):")
        for line in lines:
            print("    " + line)
        print()


if __name__ == "__main__":
    main()
