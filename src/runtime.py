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
    compCommand.append('module load rocm/6')
    compCommand.append('hipcc -fPIC -shared -g -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack therefore.cpp -o Therefore.so')
    compCommand.append('hipcc -fPIC -shared -g -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack sweep_gpu.cpp -o Sweep.so')
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

def loadSweep():
    print('loading Sweep')
    _Sweep = ctypes.cdll.LoadLibrary('./Sweep.so')
    Sweep = _Sweep.ThereforeSweep
    Sweep.restype = None
    #NP_DOUBLE_POINTER = ndpointer(DOUBLE, flags="C_CONTIGUOUS")
    Sweep.argtypes = (DOUBLE, INT)
    return(Sweep)


if (__name__ == '__main__'):

    #hardware_name = argv[1]

    dx = np.array([10, 5, 1, .5, .25, .1, .05, .01]).astype(NP_DOUBLE)
    #N_angles = np.array([4, 6, 8, 16, 32, 64, 128])
    #

    #dx = np.array([.1,.05]).astype(NP_DOUBLE)
    N = int( dx.size )

    runTimeOCI = np.zeros(N)
    runTimeSweep = np.zeros(N)

    compile()
    ThereforeOCI = loadTherefore()
    ThereforeSweep = loadSweep()

    for i in range(N):

        print("Timing Therefore at N_angles: ", dx[i])

        start = time.time()
        ThereforeOCI(dx[i], 16)
        end = time.time()
        runTimeOCI[i] = end - start

        start = time.time()
        ThereforeSweep(dx[i], 16)
        end = time.time()
        runTimeSweep[i] = end - start
        print("Total Sweep" , runTimeSweep[i])

    print("Completed")
    #dir = 'runtime_results/'
    #file_name = dir + 'runtime_' + hardware_name
    file_name = 'runtimes'
    print("Sweep")
    print(runTimeSweep)
    print("OCI")
    print(runTimeOCI)
    np.savez(file_name, OCI=runTimeOCI, Sweep=runTimeSweep, dx=dx, N_angles=24)

    