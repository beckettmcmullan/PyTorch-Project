import torch
from torch import nn

num_nodes = 4 # nodes 0 - 3 -> with edges 0->1, 0->2, 1->2, 0->3, 1->3, 2->3

class Zero(nn.Module):
    def forward(self, x):
        return 0 * x

operation_names = ['zero', 'identity', 'linear', 'nonlinear']
def make_operations(): # make fresh copies of every operation at every edge of a node
    return nn.ModuleList([
        Zero(), # index 0: output = 0
        nn.Identity(), # index 1: output = input
        nn.Linear(1,1), # index 2: output = input * weight + bias
        nn.Sequential( # index 3: learn a nonlinear relationship
            nn.Linear(1,16),
            nn.Tanh(),
            nn.Linear(16,1),
        ),
        ])

class MixedEdge(nn.Module): # 1 connection between 2 nodes
    def __init__(self):
        super().__init__()
        self.operations = make_operations() # each edge has its own weights for each operation
        self.scores = nn.Parameter(torch.zeros(len(self.operations))) # architecture scores for each operation at every edge

    def forward(self, x):
        probabilities = torch.softmax(self.scores, dim=0) # the probabilities of each operation at every edge
        return sum(probabilities[k] * op(x) for k, op in enumerate(self.operations)) # weighted sum of outputs of all operations at a edge

class Cell(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        # one MixedEdge per (i, j) pair with i < j, stored under a name like "0->2"
        self.edges = nn.ModuleDict({
            f'{i}->{j}': MixedEdge() for j in range(1, num_nodes) for i in range(j)
        })

    def forward(self, x):
        nodes = [x] # node 0 is the input
        for j in range(1, self.num_nodes):
            nodes.append(sum(self.edges[f'{i}->{j}'](nodes[i]) for i in range(j))) # node j = sum of its incoming edges
        return nodes[-1] # the last node is the output

    def architecture_parameters(self): # the scores of every edge
        return [edge.scores for edge in self.edges.values()]

    def weight_parameters(self): # the weights inside every operation
        return [p for edge in self.edges.values() for p in edge.operations.parameters()]

torch.manual_seed(0) # same result every run

x = torch.linspace(-2, 2, 32).unsqueeze(1) # 32 inputs, one feature each
y = x ** 2 + x # matching output shape

cell = Cell(num_nodes)
optimizer = torch.optim.Adam(cell.weight_parameters() + cell.architecture_parameters(), lr=0.01) # one optimizer for weights and scores, like DARTS2.py

def print_edges():
    with torch.no_grad():
        for name, edge in cell.edges.items():
            probabilities = torch.softmax(edge.scores, dim=0)
            formatted = ', '.join(f'{operation_names[k]}={p:.3f}' for k, p in enumerate(probabilities.tolist()))
            print(f'  edge {name}: {formatted}')

loss_history = [] # loss at every step, for the plots
probability_history = {name: [] for name in cell.edges} # each edge's probabilities at every step

for step in range(3000):
    optimizer.zero_grad() # clear gradients
    prediction = cell(x) # run the whole cell
    loss = nn.MSELoss()(prediction, y) # compute loss
    loss.backward() # gradients flow to every edge's weights and scores
    optimizer.step() # update weights and scores using Adam
    loss_history.append(loss.item())
    with torch.no_grad():
        for name, edge in cell.edges.items():
            probability_history[name].append(torch.softmax(edge.scores, dim=0).tolist())
    if step % 1000 == 0 or step == 2999:
        print(f'Step {step}: loss = {loss.item():.5f}')
        print_edges()

with torch.no_grad(): # pick the most likely operation on each edge, independently
    print('Chosen operation per edge:')
    for name, edge in cell.edges.items():
        best_operation = torch.argmax(edge.scores).item()
        print(f'  edge {name}: {operation_names[best_operation]}')

# ---------------------------------------------------------------- plots
# Saves the five figures explained in the README to darts/figures/03_multi_edge/
import os
import matplotlib
matplotlib.use('Agg') # write files, no window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

figure_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', '03_multi_edge')
os.makedirs(figure_dir, exist_ok=True)
colors = ['#0072B2', '#D55E00', '#E69F00', '#7B2D8E'] # one color per operation
edge_names = list(cell.edges)
final_probabilities = [torch.softmax(cell.edges[name].scores, dim=0).tolist() for name in edge_names]

# 1. training loss
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(loss_history, color=colors[0])
ax.set_yscale('log')
ax.set_xlabel('Training step')
ax.set_ylabel('MSE loss (log scale)')
ax.set_title('Training loss: how wrong the cell is over time')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(figure_dir, '1_loss.png'), dpi=150)

# 2. each edge's probabilities over training
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
for ax, name in zip(axes.flat, edge_names):
    history = torch.tensor(probability_history[name])
    for k, operation in enumerate(operation_names):
        ax.plot(history[:, k], color=colors[k], label=operation)
    ax.set_title(f'Edge {name}')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel('Step')
for ax in axes[:, 0]:
    ax.set_ylabel('Probability')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(operation_names))
fig.suptitle("How each edge's choice changes during training")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(figure_dir, '2_edge_probabilities.png'), dpi=150)

# 3. final probabilities as a heatmap
fig, ax = plt.subplots(figsize=(6.5, 5))
image = ax.imshow(final_probabilities, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i, row in enumerate(final_probabilities):
    for k, p in enumerate(row):
        ax.text(k, i, f'{p:.2f}', ha='center', va='center', color='white' if p > 0.6 else 'black')
ax.set_xticks(range(len(operation_names)), operation_names)
ax.set_yticks(range(len(edge_names)), edge_names)
ax.set_xlabel('Operation')
ax.set_ylabel('Edge')
ax.set_title('Final probability of each operation on each edge')
fig.colorbar(image)
fig.tight_layout()
fig.savefig(os.path.join(figure_dir, '3_final_heatmap.png'), dpi=150)

# 4. final cell output vs the target
with torch.no_grad():
    final_prediction = cell(x)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x.squeeze(), y.squeeze(), color='black', linewidth=2, label='Target: y = x² + x')
ax.plot(x.squeeze(), final_prediction.squeeze(), 'o', color=colors[0], label='Cell prediction')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Final cell output vs. target')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(figure_dir, '4_fit.png'), dpi=150)

# 5. the chosen operation on each edge, drawn as a graph
fig, ax = plt.subplots(figsize=(10, 5))
positions = {j: (j, 0) for j in range(num_nodes)}
curves = {1: 0.0, 2: 0.45, 3: -0.45} # straight for neighbours, curved for skips (0->3 above, the rest below)
for name, row in zip(edge_names, final_probabilities):
    i, j = map(int, name.split('->'))
    best = max(range(len(row)), key=lambda k: row[k])
    rad = curves[j - i]
    arrow = FancyArrowPatch(positions[i], positions[j], connectionstyle=f'arc3,rad={rad}',
                            arrowstyle='-|>', mutation_scale=20, color=colors[best],
                            linewidth=2 + 10 * row[best], shrinkA=22, shrinkB=22)
    ax.add_patch(arrow)
    label_y = -rad * (j - i) / 2 # midpoint of the arc
    ax.text((i + j) / 2, label_y, f'{operation_names[best]} ({row[best]:.2f})',
            ha='center', va='center', backgroundcolor='white')
for j, (px, py) in positions.items():
    role = '\n(input)' if j == 0 else '\n(output)' if j == num_nodes - 1 else ''
    ax.add_patch(plt.Circle((px, py), 0.12, color='#333333', zorder=3))
    ax.text(px, py, f'node {j}{role}', color='white', ha='center', va='center', fontsize=8, zorder=4)
ax.set_xlim(-0.4, num_nodes - 0.6)
ax.set_ylim(-0.9, 0.9)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Final cell: chosen operation on each edge (thickness = confidence)')
fig.tight_layout()
fig.savefig(os.path.join(figure_dir, '5_cell_graph.png'), dpi=150)

print(f'Saved figures to {figure_dir}')
