import numpy as np
#import matplotlib.pyplot as plt
import os
import ctypes
from numpy.ctypeslib import ndpointer
import time
from sys import argv

INT = ctypes.c_int
DOUBLE = ctypes.c_double
NP_DOUBLE = np.float64
NP_INT = np.intc

def compile(target='osx_cpu', num_threads=1):

    if target=='osx_cpu':
        compCommand = []
        compCommand .append('export OMP_NUM_THREADS={}'.format(num_threads))
        compCommand.append('g++-14 main.cpp -fpic -std=c++20 -llapack -fopenmp -o Therefore2D.so')
    
    print('compiling 2D therefore')
    for i in range(len(compCommand)):
        os.system(compCommand[i])

def loadBi():
    print('loading Therefore')
    _Therefore = ctypes.cdll.LoadLibrary('./Therefore.so')
    Therefore = _Therefore.ThereforeOCI
    Therefore.restype = DOUBLE
    #NP_DOUBLE_POINTER = ndpointer(DOUBLE, flags="C_CONTIGUOUS")
    Therefore.argtypes = (DOUBLE, DOUBLE, INT)
    return(Therefore)

if __name__ == '__main__':

    compile()
    loadBi()
