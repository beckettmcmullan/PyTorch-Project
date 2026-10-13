import torch
from torch import nn

operations = nn.ModuleList([
    nn.Identity(), # index 0: output = input
    nn.Sequential( # index 1: learn a nonlinear relationship
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    ),
])

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]]) # four inputs, one feature each
y = torch.tensor([[1.0], [8.0], [25.0], [-10.0]]) # matching output shape

scores = nn.Parameter(torch.zeros(len(operations))) # one score per operation
optimizer = torch.optim.Adam(list(operations.parameters()) + [scores], lr=0.01) # adjust each parameter seperately, including the scores for each operation (Adam)

for step in range(5000):
    optimizer.zero_grad() # clear gradients
    probabilities = torch.softmax(scores, dim=0) # turns scores into probabilities that sum to 1 whil remaining differentiable
    predictions = [op(x) for op in operations] # outputs from each operation
    prediction = sum (probabilities[i] * predictions[i] for i in range(len(operations))) # weighted sum of predictions with respect to each operators probability
    loss = nn.MSELoss()(prediction, y) # compute loss
    loss.backward() # back propagate to compute how sensitive loss function is to changes in weight, bias, and scores
    optimizer.step() # update weights using Adam
    if step % 1000 == 0 or step == 4999:
        print(f'Step {step}: loss = {loss.item()}: prediction = {prediction.tolist()}')
        print(f'Probabilities: {[probabilities[i].item() for i in range(len(operations))]}')
with torch.no_grad(): # disable gradient computation
    probabilities = torch.softmax(scores, dim=0) # calculate probabilities one more time
    best_operation = torch.argmax(probabilities).item() # find the index of the operation with the highest probability
    print(f'Best operation: {best_operation} (0 = Identity, 1 = neural network)')
