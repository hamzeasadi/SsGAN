import tensorflow as tf
import keras
import numpy as np
from matplotlib import pyplot as plt


class SSGAN():
    """This is a simplified version of semi-supervised GAN"""
    def __init__(self, inputshape, outputshape):
        self.inshp = inputshape
        self.outshp = outputshape

    # def __call__(self, *args, **kwargs):

    def __call__(self, inputshape, outputshape):
        self.inshp = inputshape
        self.outshp = outputshape

    def buildModel(self):
        pass

