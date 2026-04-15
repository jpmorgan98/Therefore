import numpy as np
import matplotlib.pyplot as plt
import therefore

file = np.load('outputs.npz')
file_azurv1 = np.load('azurv1.npz')

sim_perams = {'L': 10,
              'dt': .1,
              'offset': .1,
              'ratio': .9}

therefore.MovieMaker(file['SI'], file['OCI'], file_azurv1['azurv1'], sim_perams)

#therefore.MovieMaker(scalar_flux, scalar_flux2, L)
