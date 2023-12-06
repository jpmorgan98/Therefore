import numpy as np
import matplotlib.pyplot as plt
from mms import mms

def group2cont(x, mu, t):
    return(pow(x,2)*t + mu)

def group1cont(x,mu,t):
    return(x + t + mu)

def error(vec1, vec2):
    return( np.linalg.norm(vec1 - vec2, ord=2) )

N_angles = 24
N_cells = 170
N_groups = 2
N_time = 1

dt = 1.0

file_name_base = 'afluxUnsorted'
file_name_base2 = 'mms_sol'
file_ext = '.csv'

[angles, weights] = np.polynomial.legendre.leggauss(N_angles)

# matrix order of the whole ass problem
SIZE_problem = N_cells*N_angles*N_groups*4
# size of the cell blocks in all groups and angle
SIZE_cellBlocks = N_angles*N_groups*4
# size of the group blocks in all angle within a cell
SIZE_groupBlocks = N_angles*4
# size of the angle blocks within a group and angle
SIZE_angleBlocks = 4

x = np.genfromtxt('x.csv', dtype=np.float64, delimiter=',', skip_header=1)
x = x[:,0]
dx = x[1]

af_wp = np.zeros((N_time*2, N_groups, N_angles, 2*N_cells))
sf_wp = np.zeros((N_time*2, N_groups, 2*N_cells))
assert (int(af_wp.size/N_time) == SIZE_problem)

af_mms = np.zeros((N_time*2, N_groups, N_angles, 2*N_cells))
sf_mms = np.zeros((N_time*2, N_groups, 2*N_cells))

af_mms_cont = np.zeros((N_time*2, N_groups, N_angles, 2*N_cells))
sf_mms_cont = np.zeros((N_time*2, N_groups, 2*N_cells))

# schuky-duck the angular flux together
for t in range(N_time):
    # import csv file 
    file = file_name_base+str(t)+file_ext
    af_raw = np.genfromtxt(file, dtype=np.float64, delimiter=',', skip_header=2)
    af_raw = af_raw[:,0]

    # import mms data
    file2 = file_name_base2+str(t)+file_ext
    mms_raw = np.genfromtxt(file2, dtype=np.float64, delimiter=',', skip_header=2)
    mms_raw = mms_raw[:,0]

    if (af_raw.size != SIZE_problem):
        print(">>>ERROR<<<")
        print("Shape mismatch")
        print("af_raw shape: {0}".format(af_raw.size))
        print("SIZE_problem: {0}".format(SIZE_problem))
        #assert (af_raw.size == SIZE_problem)

    for i in range(N_cells):
        for g in range(N_groups):
            for n in range(N_angles):
                index_start = (i*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*n)

                af_wp[t*2  ,g,n,2*i]   = af_raw[index_start]
                af_wp[t*2  ,g,n,2*i+1] = af_raw[index_start+1]
                af_wp[t*2+1,g,n,2*i]   = af_raw[index_start+2]
                af_wp[t*2+1,g,n,2*i+1] = af_raw[index_start+3]

                sf_wp[t*2  ,g,2*i]   += weights[n] * af_raw[index_start]
                sf_wp[t*2  ,g,2*i+1] += weights[n] * af_raw[index_start+1]
                sf_wp[t*2+1,g,2*i]   += weights[n] * af_raw[index_start+2]
                sf_wp[t*2+1,g,2*i+1] += weights[n] * af_raw[index_start+3]

                '''
                af_mms[t*2  ,g,n,2*i]   = mms_raw[index_start]
                af_mms[t*2  ,g,n,2*i+1] = mms_raw[index_start+1]
                af_mms[t*2+1,g,n,2*i]   = mms_raw[index_start+2]
                af_mms[t*2+1,g,n,2*i+1] = mms_raw[index_start+3]

                sf_mms[t*2  ,g,2*i]   += weights[n] * mms_raw[index_start]
                sf_mms[t*2  ,g,2*i+1] += weights[n] * mms_raw[index_start+1]
                sf_mms[t*2+1,g,2*i]   += weights[n] * mms_raw[index_start+2]
                sf_mms[t*2+1,g,2*i+1] += weights[n] * mms_raw[index_start+3]

                mms_cont_raw = np.zeros(4)
                if (g==0):
                    mms_cont_raw[0] = group1cont(x[i*2], angles[n], t*dt  )
                    mms_cont_raw[1] = group1cont(x[i*2+1], angles[n], dt*1)
                    mms_cont_raw[2] = group1cont(x[i*2], angles[n], t*dt+.5*dt)
                    mms_cont_raw[3] = group1cont(x[i*2+1], angles[n], t*dt+.5*dt)
                elif (g==1):
                    mms_cont_raw[0] = group2cont(x[i*2], angles[n], t*dt  )
                    mms_cont_raw[1] = group2cont(x[i*2+1], angles[n], dt*1)
                    mms_cont_raw[2] = group2cont(x[i*2], angles[n], t*dt+.5*dt)
                    mms_cont_raw[3] = group2cont(x[i*2+1], angles[n], t*dt+.5*dt)
                #print(mms_cont_raw)

                af_mms_cont[t*2  ,g,n,2*i]   = mms_cont_raw[0]
                af_mms_cont[t*2  ,g,n,2*i+1] = mms_cont_raw[1]
                af_mms_cont[t*2+1,g,n,2*i]   = mms_cont_raw[2]
                af_mms_cont[t*2+1,g,n,2*i+1] = mms_cont_raw[3]

                sf_mms_cont[t*2  ,g,2*i]   += weights[n] * mms_cont_raw[0]
                sf_mms_cont[t*2  ,g,2*i+1] += weights[n] * mms_cont_raw[1]
                sf_mms_cont[t*2+1,g,2*i]   += weights[n] * mms_cont_raw[2]
                sf_mms_cont[t*2+1,g,2*i+1] += weights[n] * mms_cont_raw[3]


print(error(sf_wp[2,0,:], sf_mms[2,0,:]))
'''
#print(sf_mms_cont[0,0,:])

temp = np.zeros(2*N_cells)
for i in range(N_cells*2):
    for n in range(N_angles):
        temp[i] += weights[n] * group1cont(x[i], angles[n], 0)

#print(temp == sf_mms_cont[0,0,:])

#print(sf_mms)
plt.figure()
#plt.plot(x, af_wp [0,1, 0,:], 'k', label='computed')
#plt.plot(x, af_mms[0,0, 0,:], 'k*', label='mms')
#plt.plot(x, af_wp [0,1, 1,:], 'r', label='computed')
#plt.plot(x, af_mms[0,0, 1,:], 'r*', label='mms')
#plt.plot(x, sf_wp [0,1,:], 'g', label='computed')
#plt.plot(x, sf_mms[0,0,:], 'g*', label='mms')
#plt.plot(x, af_mms[0,1, 1,:], 'r*', label='mms')
#plt.plot(x, sf_wp[0,1,:], 'g', label='computed')


#plt.plot(x, temp, '+', label='cont')
plt.plot(x, sf_wp[0,0,:], label='g1 -- no source')
#plt.plot(x[:,0], sf_wp[5,0,:], label='g1 -- no source')
#plt.plot(x[:,0], sf_wp[5,1,:], label='g1 -- no source')
#plt.plot(x[:,0], sf_wp[7,0,:], label='g1 -- no source')
#plt.plot(x[:,0], sf_wp[7,1,:], label='g1 -- no source')
plt.xlabel('Distance')
plt.ylabel('Sc Fl')
plt.title('Single region -- trouble shoot time step=1')
plt.legend()
plt.savefig('_soultion.png')


