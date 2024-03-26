#!/usr/bin/env python
import argparse
import os
import sys
import traceback
import time
import warnings
import pickle
from collections import OrderedDict
import yaml
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


#weights_path='./work_dir/recognition/Actiondata2/20batchsize/epoch150_model.pt'
weights_path='model/self-attention_1_94.2529'
weights = torch.load(weights_path)
score=weights['prec_score']
label=weights['target_vector']
#weights = OrderedDict([[k.split('module.')[-1],v.cpu()] for k, v in weights.items()])
print("score:",score)
print("label:",label)
