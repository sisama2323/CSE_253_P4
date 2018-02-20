import numpy as np
import os.path as osp
import os
import sys
from util import music


input_file = 'CSE_253_P4/data/input.txt'
# sample_music = 'data/sample-music.txt'


music_book = []
sheet = []
with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        # check the first 7 letter
        if line[0:7] == '<start>':
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
            music_book[-1].tune = line[2:-4]
        elif line[0] == 'M':
            music_book[-1].beats = line[2:-4]
        elif line[0] == 'L':
            music_book[-1].note_len = line[2:-4]
        elif line[0] == 'K':
            music_book[-1].key = line[2:-4]
        elif line[0] == 'L':
            music_book[-1].note_len = line[2:-4]
        # store body
        elif line[1] != ':' and line[0] != '<' and line[0] != '<':
            body += line[:-4]
        elif line[0:5] == '<end>':
            music_book[-1].body = body
            sheet.append(body)


a=5

