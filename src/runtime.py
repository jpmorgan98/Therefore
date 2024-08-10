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

file_name = 'runtimes'


def compile():
    compCommand = []
    compCommand.append('hipcc -fPIC -shared -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack therefore.cpp -o Therefore.so')
    compCommand.append('hipcc -fPIC -shared -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack sweep_gpu.cpp -o Sweep.so')
    #-fopenmp
    print('compiling therefore and sweep')

    for i in range(len(compCommand)):
        os.system(compCommand[i])

def loadTherefore():
    print('loading Therefore')
    _Therefore = ctypes.cdll.LoadLibrary('./Therefore.so')
    Therefore = _Therefore.ThereforeOCI
    Therefore.restype = DOUBLE
    #NP_DOUBLE_POINTER = ndpointer(DOUBLE, flags="C_CONTIGUOUS")
    Therefore.argtypes = (DOUBLE, INT)
    return(Therefore)

def loadSweep():
    print('loading Sweep')
    _Sweep = ctypes.cdll.LoadLibrary('./Sweep.so')
    Sweep = _Sweep.ThereforeSweep
    Sweep.restype = DOUBLE
    #NP_DOUBLE_POINTER = ndpointer(DOUBLE, flags="C_CONTIGUOUS")
    Sweep.argtypes = (DOUBLE, INT)
    return(Sweep)


if (__name__ == '__main__'):

    #hardware_name = argv[1]

    # first value is repeated to allow codes to spool up
    # this time will be removed and not shown
    dx = np.array([10, 10, 5, 1, .5, .75, .25, .15, .1, .05, .01]).astype(NP_DOUBLE)
    angles = np.array([4, 8, 16, 32]).astype(NP_INT)

    #dx = np.array([10]).astype(NP_DOUBLE)
    #angles = np.array([4]).astype(NP_INT)

    #dx = np.array([.1,.05]).astype(NP_DOUBLE)
    N_space = int( dx.size )
    N_angles = int ( angles.size )

    runTimeOCI = np.zeros((N_space, N_angles))
    runTimeSweep = np.zeros((N_space, N_angles))

    compile()
    ThereforeOCI = loadTherefore()
    ThereforeSweep = loadSweep()

    for j in range(N_angles):
        for i in range(N_space):
            
            print()
            print(">>>Timing Therefore at dx: ", dx[i], " and N: ", angles[j])
            print()

            #start = time.time()
            time_oci = ThereforeOCI(dx[i], angles[j])
            #end = time.time()
            runTimeOCI[i, j] = time_oci# end - start
            print("     Total OCI" , runTimeOCI[i, j])

            #start = time.time()
            time_sweep = ThereforeSweep(dx[i], angles[j])
            #end = time.time()
            runTimeSweep[i, j] = time_sweep#end - start
            print("     Total Sweep" , runTimeSweep[i, j])

            np.savez(file_name, OCI=runTimeOCI, Sweep=runTimeSweep, dx=dx, angles=angles, i=i, j=j)

    print("Completed")
    #dir = 'runtime_results/'
    #file_name = dir + 'runtime_' + hardware_name
    file_name = 'runtimes'
    print("Sweep")
    print(runTimeSweep)
    print("OCI")
    print(runTimeOCI)
    np.savez(file_name, OCI=runTimeOCI, Sweep=runTimeSweep, dx=dx, angles=angles)