import torch as torch
import dlc_practical_prologue as prologue
import progressbar
from dlc_practical_prologue import *
import matplotlib.pyplot as plt
import numpy as np
import math
# Generator Code

class Neural_Network:
  def __init__(self, hidden_size, in_size=3072, out_size=5, lr=0.001, std=1e-9):
    self.W1 = torch.randn((in_size, hidden_size)).normal_(0, std)
    self.b1 = torch.randn((1,hidden_size)).normal_(0, std)
    self.W2 = torch.randn((hidden_size,out_size)).normal_(0, std)
    self.b2 = torch.randn((1,out_size)).normal_(0, std)

    self.dW1 = torch.zeros((in_size, hidden_size))
    self.db1 = torch.zeros((1,hidden_size))
    self.dW2 = torch.zeros((hidden_size,out_size))
    self.db2 = torch.zeros((1,out_size))

    self.lr = lr

  def forward(self,x):
    x_1 = x @ self.W1 + self.b1
    s_1 = sigmoid(x_1)
    x_2 = s_1 @ self.W2 + self.b2
    out =  sigmoid(x_2)
    return x_1, s_1, x_2, out

  def backward_pass(self, t, x, s1, x1,s2, x2):
    self.dW2 = s1.T @ (dloss(s2,t) * dsigmoid(x2)) 
    self.db2 = dloss(s2,t) * dsigmoid(x2)
    self.dW1 = x.T @ ((dloss(s2,t)*dsigmoid(x2)@ self.W2.T) * dsigmoid(x1))
    self.db1 = (dloss(s2,t)*dsigmoid(x2)@ self.W2.T) * dsigmoid(x1)

    self.W1 -= self.lr * self.dW1
    self.b1 -= self.lr * self.db1
    self.W2 -= self.lr * self.dW2
    self.b2 -= self.lr * self.db2


  def zero_gradient(self):
    self.dW1.zero_()
    self.db1.zero_()
    self.dW2.zero_()
    self.db2.zero_()


  def train(self, X, y,test_input,test_target,  epoch):
    test_losses1 = []
    train_losses1 = []
    for i in range(epoch):
      epoch_loss = 0.0
      for j, data in enumerate(X):
        ## forward
        data = data[None,...]
        target = y[j][None,...]
        x_1, s_1, x_2, out = self.forward(data)

        t = target  
        epoch_loss += loss(out, t).item() ## loss

        ## backward
        self.backward_pass(t, data, s_1, x_1, out, x_2)

        ## zero gradient
        self.zero_gradient()

      if i % 100 == 0:
        test_loss = loss(model1.forward(test_input)[-1], test_target)
        test_losses1.append(test_loss)
        train_loss = loss(model1.forward(train_input)[-1], train_target)
        train_losses1.append(train_loss)
        print(f'Epoch {i} Train Loss: {epoch_loss/ X.shape[0]} and Test loss: {test_loss / test_input.shape[0]}')
    return (train_losses1, test_losses1)

  def accuracy(self,pred,target):
    pred_ones = torch.argmax(pred, dim=1)
    target_ones = torch.nonzero(target)[:, 1]
    return ((pred_ones == target_ones).sum().item() / target.shape[0])