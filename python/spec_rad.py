"""
Created on Sun May 15 20:34:03 2022

@author: jacksonmorgan
"""

import numpy as np
import matplotlib.pyplot as plt
import therefore
from timeit import default_timer as timer

def flatLinePlot(x, y):
    for i in range(y.size):
        xx = x[i:i+2]
        yy = [y[i], y[i]]
        plt.plot(xx, yy, '-k')

data_type = np.float64

L = 10
xsec = 0.5
source = 1
v = 2

#dx = np.linspace(.01, 2, 75)
dx = .5

ratio = np.linspace(0, 1, 100)

mfp = np.linspace(0, 20, 25)
dt = np.linspace(0.01,0.4,100)

x = ratio
y = mfp

xs = x.size
ys = y.size

epsilon = 1e-16

no_converge_oci = np.zeros([xs, ys])
spec_rad_oci = np.zeros([xs, ys])
no_converge_si = np.zeros([xs, ys])
spec_rad_si = np.zeros([xs, ys])

total_runs = xs * ys

time_oci = 0
time_si = 0

for i in range(xs):
    for k in range(ys):
        
        print('Percent done: %2d' %(((i*ys+k)/total_runs)*100),end='\r')
        
        #xsec_hat = xsec + 1/(v*dt[k])
        xsec = mfp[k]/dx
        scattering_xsec = xsec*ratio[i]
        
        N_mesh = int(L/dx)

        dx_mesh = dx*np.ones(N_mesh, data_type)
        xsec_mesh = xsec*np.ones(N_mesh, data_type)
        xsec_scatter_mesh = scattering_xsec*np.ones(N_mesh, data_type)
        source_mesh = source*np.ones(N_mesh, data_type)

        sim_perams = {'data_type': data_type,
                      'N_angles': 2,
                      'L': L,
                      'N_mesh': N_mesh,
                      'boundary_condition_left': 'reflecting',
                      'boundary_condition_right': 'reflecting',
                      'left_in_mag': 10,
                      'right_in_mag': 10,
                      'left_in_angle': .3,
                      'right_in_angle': -.3,
                      'max loops': 10000,
                      'tolerance': epsilon}

        start = timer()
        [sf, cur, spec_rad_oci[i,k], no_converge_oci[i,k], loops] = therefore.OCI(sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh)
        time_oci += timer() - start
        start = timer()
        [sf, cur, spec_rad_si[i,k], no_converge_si[i,k], loops] = therefore.SourceItteration(sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh)
        time_si += timer() - start

if ((no_converge_si == False).any):
    np.set_printoptions(linewidth=np.inf)
    print()
    print('>>>WARNING: Some runs of SI did not converge before itter kick.')
    print('            Recomend grow max_itter value and run again  <<<')
    print('')
    print(spec_rad_si)
    print()

if ((no_converge_oci == False).any):
    np.set_printoptions(linewidth=np.inf)
    print()
    print('>>>WARNING: Some runs of OCI did not converge before itter kick.')
    print('            Recomend grow max_itter value and run again  <<<')
    print('')
    #print(spec_rad_oci)
    print()

print()
print('Total time for OCI computations: {0}'.format(time_oci))
print('Total time for SI computations:  {0}'.format(time_si))
print()


#mfp = xsec*dx
#mfp = xsec*dx
#np.savez('spectral_radius_s2', mfp=mfp, ratio=ratio, spec_rad_oci=spec_rad_oci, spec_rad_si=spec_rad_si)

[Xx, Yy] = np.meshgrid(y,x)

N_si = np.zeros([xs,ys])
N_oci = np.zeros([xs,ys])

for i in range(xs):
    for k in range(ys):
        N_si[i,k] = np.log(epsilon)/np.log(spec_rad_si[i,k])
        N_oci[i,k] = np.log(epsilon)/np.log(spec_rad_oci[i,k])

N_ratio = N_si/N_oci

spec_rad_ratio = spec_rad_si/spec_rad_oci

np.savez('spec_rad', spec_rad_si=spec_rad_si, spec_rad_oci=spec_rad_oci, mfp=mfp, ratio=ratio, spec_rad_ratio=spec_rad_ratio, N_ratio=N_ratio)



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xx,Yy,spec_rad_oci, cmap='viridis')
plt.title('OCI SCB Spectral Radius Plot')
plt.xlabel(r'mfp [$\sigma * \Delta x$]')
plt.ylabel('Scattering Ratio [$Σ_s$/Σ]')
ax.set_zlabel('Spectrial Radius [ρ]')
plt.savefig('specrad_oci',dpi=600)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xx,Yy,spec_rad_si, cmap='viridis')
plt.title('SI SCB Spectral Radius Plot')
plt.xlabel(r'mfp [$\sigma * \Delta x$]')
plt.ylabel('Scattering Ratio [$Σ_s$/Σ]')
ax.set_zlabel('Spectrial Radius [ρ]')
plt.savefig('specrad_si',dpi=600)

#plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xx,Yy,spec_rad_si)
surf = ax.plot_surface(Xx,Yy,spec_rad_oci)
plt.title('SI SCB Spectral Radius Plot')
plt.xlabel(r'$\Delta t$')
plt.ylabel('Scattering Ratio [$Σ_s$/Σ]')
ax.set_zlabel('Spectrial Radius [ρ]')
plt.savefig('specrad_both',dpi=600)


#plt.show()


#print(N_ratio)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xx,Yy,spec_rad_ratio, cmap='coolwarm')
plt.title('Ratio')
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'Scattering Ratio [$\sigma_s/\sigma$]')
ax.set_zlabel(r'$\frac{N_{SI}}{N_{OCI}}$')


plt.show()