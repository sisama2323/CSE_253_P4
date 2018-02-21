from __future__ import unicode_literals, print_function, division
import numpy as np 
import unicodedata
from io import open
import glob
import string
import torch

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


class music():
    def __init__(self, Body='', X=0, T='', R='', C='', A='', M=1, L=1./4., K='D'):
        self.ref_num = X
        self.tune = T
        self.rhy = R
        self.comp = C
        self.area = A
        self.beats = M
        self.note_len = L
        self.key = K
        self.body = Body

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor