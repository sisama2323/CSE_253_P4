import numpy as np
import os.path as osp
import os
import sys
from util import *
import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F




input_file = 'CSE_253_P4/data/input.txt'
# input_file = 'data/input.txt'
# sample_music = 'data/sample-music.txt'

# music_book contain all the information about every song, including beats, name, ryh
music_book = []
# contain only the body of music
sheet = []
with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        # check the first 7 letter
        if len(line) <= 3:
            pass
        elif line[0:7] == '<start>':
            # initialize a music
            music_book.append(music())
            # initialize body
            body = ''
        elif line[0] == 'X':
            try:
                music_book[-1].ref_num = int(line[2])
            except ValueError:
                music_book[-1].ref_num = line[2]
        elif line[0] == 'T':
            music_book[-1].tune = line[2:-1]
        elif line[0] == 'M':
            music_book[-1].beats = line[2:-1]
        elif line[0] == 'L':
            music_book[-1].note_len = line[2:-1]
        elif line[0] == 'K':
            music_book[-1].key = line[2:-1]
        elif line[0] == 'L':
            music_book[-1].note_len = line[2:-1]
        # store body
        elif line[1] != ':' and line[0] != '<' and line[0] != '<':
            body += line[:-1]
        elif line[0:5] == '<end>':
            music_book[-1].body = body
            sheet.append(body)



for piece in sheet:
    lineToTensor(piece)

input = Variable(lineToTensor(piece))
# train_input = trainset.train_data[train_idx, :, :, :]
# train_labels = [trainset.train_labels[j] for j in train_idx]

# val_input = trainset.train_data[val_idx, :, :, :]
# val_labels = [trainset.train_labels[j] for j in val_idx]
# # prepare for training set and validation set
# train_set = torch.utils.data.TensorDataset(torch.from_numpy(train_input.reshape(44998, 3, 32, 32)), torch.from_numpy(np.array(train_labels)))
# val_set = torch.utils.data.TensorDataset(torch.from_numpy(val_input.reshape(5000, 3, 32, 32)), torch.from_numpy(np.array(val_labels)))

# trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
# validationloader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)

a=5