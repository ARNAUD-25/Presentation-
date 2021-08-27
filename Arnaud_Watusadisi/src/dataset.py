
import torch as torch
import dlc_practical_prologue as prologue
import progressbar
from dlc_practical_prologue import *
import matplotlib.pyplot as plt
import numpy as np
import math

# We use cifar 10 like data.
train_input, train_target, test_input, test_target = load_data(cifar = True, one_hot_labels =True, normalize =True, flatten = True)