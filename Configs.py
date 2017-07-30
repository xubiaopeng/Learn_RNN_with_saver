#!/usr/bin/env python
#coding=utf-8
"""
@version: 1.0
@author: jin.dai
@license: Apache Licence
@contact: daijing491@gmail.com
@software: PyCharm
@file: Configs.py.py
@time: 2017/7/15 11:33
@description: class definition of configuration parameters
"""
#import os

class Config:
    def __init__(self):
        # Hyperparameters
        self.BATCH_SIZE = 50  ## number of proteins
        self.LR = 0.006  ## learning rate
        # Network Parameters
        self.MAX_STEPS = 200  ## number of residues
        self.INPUT_SIZE = 2   ## number of features for input, such as MSA information ...
        self.OUTPUT_SIZE = 2  ## number of labels for output, such as Ramachandran angles ...
        self.CELL_SIZE = 64   ## size of cell


class ProteinConfig:
    def __init__(self):
        # Hyperparameters
        self.BATCH_SIZE = 16  ## number of proteins
        self.LR = 0.01  ## learning rate
        # Network Parameters
        self.MAX_STEPS = 300  ## number of residues
        self.INPUT_SIZE = 25   ## number of features for input, such as MSA information ...
        self.OUTPUT_SIZE = 3  ## number of labels for output, such as Ramachandran angles ...
        self.CELL_SIZE = 128   ## size of cell
        self.NUM_LAYERS = 3 ## number of LSTM layers
        self.checkpoint_dir = "./result/"
