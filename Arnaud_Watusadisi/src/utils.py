import torch as torch
import dlc_practical_prologue as prologue
import progressbar
from dlc_practical_prologue import *
import matplotlib.pyplot as plt
import numpy as np
import math

# custom weights initialization called on netG and netD
def loss(v,t):
  return torch.sum((t - v).pow(2))

def dloss(v, t):
  return  - 2 * (t - v)

def compute_error(pred,target):
    pred_ones = torch.argmax(pred, dim=1)
    target_ones = torch.nonzero(target)[:, 1]
    return ((pred_ones != target_ones).sum().item() / target.shape[0])

def sigmoid(x):
  s=1/(1+torch.exp(-x))
  return s

def dsigmoid(x):
  s=1/(1+torch.exp(-x))
  ds=s*(1-s)  
  return ds

def B_sigmoid(x):
  s=(1-torch.exp(-x))/(1+torch.exp(-x))
  return s

def dB_sigmoid(x):
  s=(2*torch.exp(-x))/((1+torch.exp(-x)).pow(2))
  ds= (s*(1-s))  
  return ds

  def sigma(x):
  return torch.tanh(x)

  def dsigma(x):
  return 1 - torch.tanh(x)**2

        
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