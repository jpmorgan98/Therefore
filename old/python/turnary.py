from tkinter import X
import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import therefore
from timeit import default_timer as timer
import ternary

#import mcdc
import numpy as np
import h5py

N_angles = np.array([256])

scale = 15

data_type = np.float64

L = 10
xsec = 2
source_mat = 0.1
v = 4
N_time = 5

ratios = np.linspace(0.0, 0.95, scale+1)
mfps = np.linspace(0.05, 10.0, scale+1)
dts = np.linspace(0.05, 1.0, scale+1)

N_mesh = 5
max_time = 5
dt = 5
sim_perams = {'data_type': data_type,
              'N_angles': 2,
              'L': L,
              'N_mesh': N_mesh,
              'boundary_condition_left':  'vacuum',
              'boundary_condition_right': 'vacuum',
              'left_in_mag': 0.1,
              'right_in_mag': 0.3,
              'left_in_angle': 0,
              'right_in_angle': 0,
              'max loops': 10000,
              'velocity': v,
              'dt': dt,
              'max time': max_time,
              'N_time': N_time,
              'offset': 0,
              'tolerance': 1e-9,
              'print': False}


print()
print('THEREFORE RUNTIME TEST OF GPU BACKEND')
print(ratios)
print(mfps)
print(dts)
print()

for p in range(N_angles.size):
    
    print()
    print()
    print('><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><')
    print('>>>Angle: {}<<<'.format(N_angles[p]))
    print('><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><')
    print()
    print()

    N_angle = N_angles[p]

    runtimes_oci = np.zeros((136, 4))
    runtimes_si = np.zeros((136, 4))
    runtime_ratio = np.zeros((136, 4))

    itter = 0
    for (i, j, k) in ternary.helpers.simplex_iterator(scale):
        
        print('{} out of {}'.format(itter, 135))

        ratio = ratios[i]
        mfp = mfps[j]
        dt = dts[k]

        dx = mfp/xsec
        N_mesh = int(L/dx)

        N_ans = int(2*N_mesh)
        inital_angular_flux = np.zeros([N_angle, N_ans], data_type)


        scattering_xsec = ratio * xsec
        xsec_scatter_mesh = scattering_xsec*np.ones(N_mesh, data_type)

        dx_mesh = dx*np.ones(N_mesh, data_type)

        sim_perams['dt'] = dt
        sim_perams['max_time'] = dt * N_time
        sim_perams['N_mesh'] = N_mesh
        sim_perams['N_angles'] = N_angle

        xsec_mesh = xsec*np.ones(N_mesh, data_type)
        source_mesh = source_mat*np.ones([N_mesh], data_type)

        [sfMB, current, spec_rads, run_oci] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'Big') #OCI_MB_GPU
        runtimes_oci[itter, 0] = i
        runtimes_oci[itter, 1] = j
        runtimes_oci[itter, 2] = k
        runtimes_oci[itter, 3] = run_oci


        [sfMBSi_gpu, current, spec_rads, run_si] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI_MB_GPU') #OCI_MB_GPU
        runtimes_oci[itter, 0] = i
        runtimes_oci[itter, 1] = j
        runtimes_oci[itter, 2] = k
        runtimes_oci[itter, 3] = run_si

        runtime_ratio[itter, 0] = i
        runtime_ratio[itter, 1] = j
        runtime_ratio[itter, 2] = k
        runtime_ratio[itter, 3] = run_si / run_oci

        print()
        print('>>>Cycle {} out of {}<<<'.format(itter, 135))
        print('=======================================================')
        print('c            {}'.format(ratio))
        print('mfp          {}'.format(mfp))
        print('dt           {}'.format(dt))
        print('OCI Run      {}'.format(run_oci))
        print('SI Run       {}'.format(run_si))
        print('SI/OCI       {}'.format(run_si / run_oci))
        print()
        print()

        itter += 1

    
    np.savez('local_runs2/runtimes_{}.npz'.format(N_angle), runtimes_oci=runtimes_oci, runtimes_si=runtimes_si, runtime_ratio=runtime_ratio, scale=scale, raios=ratios, mfps=mfps, dts=dts)


'''
data = dict()
for p in range(runtime_ratio.shape[0]):
    data[(runtime_ratio[p,0], runtime_ratio[p,1], runtime_ratio[p,2])] = runtime_ratio[p,3]


figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 8)
tax.heatmap(data=data, cmap=None)
tax.boundary(linewidth=2.0)
tax.set_title("Wall-clock Runtime Ratio ($\Delta t_{OCI}/\Delta t_{SI}$) in $S_2$ on GPU")

tax.set_axis_limits({'b': [dts[0], dts[-1]], 'l': [ratios[0], ratios[-1]], 'r': [mfps[0], mfps[-1]]})
tick_formats = {'b': "%.2f", 'l': "%.2f", 'r': "%d"}

tax.get_ticks_from_axis_limits()
tax.set_custom_ticks(fontsize=10, offset=0.02, tick_formats=tick_formats)
tax.get_axes().axis('off')

tax.left_axis_label("scattering ratio [$\Sigma_s/\Sigma$]", offset=0.13)
tax.right_axis_label("mfp thickness [$\Sigma*\Delta x$]", offset=0.13)
tax.bottom_axis_label("$\Delta t$", offset=0.0)

tax.clear_matplotlib_ticks()
tax._redraw_labels()
plt.tight_layout()
tax.show()
'''
