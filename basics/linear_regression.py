import torch
from torch import nn

model = nn.Linear(1, 1) # prediction = x * weight + bias
x = torch.tensor([2.0]) # input to the network
y = torch.tensor([4.0])  # target output

loss_function = nn.MSELoss() # mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent

for step in range(100):
    optimizer.zero_grad() # clear previous gradients
    prediction = model(x) # make another guess
    loss = loss_function(prediction, y) # compute loss
    loss.backward() # back propagate to compute how sensitive loss function is to changes in weight and bias
    optimizer.step() # update weights using SGD

    if step % 10 == 0:
        print(f'Step {step}: loss = {loss.item()}: prediction = {prediction.item()}')
