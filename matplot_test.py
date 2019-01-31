from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously

import matplotlib.pyplot as plt
import matplotlib as mpl# ; #mpl.use("Qt4Agg")
import matplotlib.animation as manimation

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import gym, os, sys, time, argparse
sys.path.append('..')
from visualize_atari import *

plt.plot([0,1,2],[0,1,2])
plt.show()