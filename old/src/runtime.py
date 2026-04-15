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
    compCommand.append('hipcc -fPIC -O3 -shared -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack therefore.cpp -o Therefore.so')
    compCommand.append('hipcc -fPIC -O3 -shared -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack sweep_gpu.cpp -o Sweep.so')
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
    Therefore.argtypes = (DOUBLE, DOUBLE, INT)
    return(Therefore)

def loadSweep():
    print('loading Sweep')
    _Sweep = ctypes.cdll.LoadLibrary('./Sweep.so')
    Sweep = _Sweep.ThereforeSweep
    Sweep.restype = DOUBLE
    #NP_DOUBLE_POINTER = ndpointer(DOUBLE, flags="C_CONTIGUOUS")
    Sweep.argtypes = (DOUBLE, DOUBLE, INT)
    return(Sweep)


if (__name__ == '__main__'):

    #hardware_name = argv[1]

    # first value is repeated to allow codes to spool up
    # this time will be removed and not shown
    #dx = np.array([10, 10, 5, 1, .75, .5, .25, .15, .1, .075, .05, .01]).astype(NP_DOUBLE)
    #angles = np.array([4, 8, 16, 32]).astype(NP_INT)

    #dx = np.array([5]).astype(NP_DOUBLE)
    #angles = np.array([2]).astype(NP_INT)

    dx = np.array([1]).astype(NP_DOUBLE)
    dt = np.array([10, 5, 1.0, .5, .1, .05, .01, .005, .001]).astype(NP_DOUBLE)
    angles = np.array([8]).astype(NP_INT)

    #dx = np.array([.1,.05]).astype(NP_DOUBLE)
    N_space = int( dx.size )
    N_angles = int ( angles.size )
    N_time = int( dt.size )

    runTimeOCI = np.zeros((N_angles, N_space, N_time))
    runTimeSweep = np.zeros((N_angles, N_space, N_time))

    compile()
    ThereforeOCI = loadTherefore()
    ThereforeSweep = loadSweep()

    for k in range(N_angles):
        for j in range(N_time):
            for i in range(N_space):
                
                print()
                print(">>>Timing Therefore at dx: ", dx[i], " dt: ", dt[j], " and N: ", angles[k])
                print()

                #start = time.time()
                time_oci = ThereforeOCI(dx[i], dt[j], angles[k])
                #end = time.time()
                runTimeOCI[k, i, j] = time_oci# end - start
                print("     Total OCI" , runTimeOCI[k, i, j])

                #start = time.time()
                time_sweep = ThereforeSweep(dx[i], dt[j], angles[k])
                #end = time.time()
                runTimeSweep[k, i, j] = time_sweep#end - start
                print("     Total Sweep" , runTimeSweep[k, i, j])

                np.savez(file_name, OCI=runTimeOCI, Sweep=runTimeSweep, dx=dx, dt=dt, angles=angles, i=i, j=j)

    print("Completed")
    #dir = 'runtime_results/'
    #file_name = dir + 'runtime_' + hardware_name
    file_name = 'runtimes2'
    print("Sweep")
    print(runTimeSweep)
    print("OCI")
    print(runTimeOCI)
    np.savez(file_name, OCI=runTimeOCI, Sweep=runTimeSweep, dx=dx, angles=angles)