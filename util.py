import numpy as np 


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