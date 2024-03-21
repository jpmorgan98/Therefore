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

def compile():
    compCommand = []
    compCommand.append('module load rocm/6 lapack')
    compCommand.append('hipcc -fPIC -shared -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -L/home/joamorga/miniconda3/lib -lrocsolver -lrocblas -llapack therefore.cpp -o Therefore.so')
    #-fopenmp
    print('compiling therefore')

    for i in range(len(compCommand)):
        os.system(compCommand[i])

def loadTherefore():
    print('loading Therefore')
    _Therefore = ctypes.cdll.LoadLibrary('./Therefore.so')
    Therefore = _Therefore.ThereforeOCI
    Therefore.restype = None
    #NP_DOUBLE_POINTER = ndpointer(DOUBLE, flags="C_CONTIGUOUS")
    Therefore.argtypes = (DOUBLE, INT)
    return(Therefore)




if (__name__ == '__main__'):

    hardware_name = argv[1]

    dx = np.array([.5]).astype(NP_DOUBLE)
    #dx = np.array([.1, .075, .05, .025, .01, .0075, .005, .0025, .001]).astype(NP_DOUBLE)
    #N_angles = np.array([4, 6, 8, 16, 32, 64, 128])

    #dx = np.array([.1,.05]).astype(NP_DOUBLE)
    N = int( dx.size )

    runTime = np.zeros(N)

    compile()
    Therefore = loadTherefore()

    for i in range(N):

        print("Timing Therefore at N_angles: ", dx[i])

        start = time.time()
        Therefore(dx[i], 4)
        end = time.time()
        runTime[i] = end - start
        print(runTime[i])

    dir = 'runtime_results/'
    file_name = dir + 'runtime_' + hardware_name
    print(runTime)
    np.savez(file_name, runtime=runTime, dx=dx, N_angles=24)

    