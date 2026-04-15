import numpy as np

import therefore.src.utilities as utl
import therefore.src.itterationSchemes as ss

from timeit import default_timer as timer

def euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source, backend, theta=1.0):
    velocity = sim_perams['velocity']
    dt = float(sim_perams['dt'])
    N_time = sim_perams['N_time']
    N_angles = sim_perams['N_angles']
    N_mesh = sim_perams['N_mesh']
    data_type = sim_perams['data_type']
    printer = sim_perams['print']
    
    N_ans = int(2*N_mesh)
    angular_flux_total = np.zeros([N_angles, N_ans, N_time], data_type)
    angular_flux_last = np.zeros([N_angles, N_ans], data_type)
    current_total = np.zeros([N_ans, N_time], data_type)
    scalar_flux = np.zeros([N_ans, N_time+1], data_type)
    spec_rad = np.zeros(N_time)

    [angles, weights] = np.polynomial.legendre.leggauss(N_angles)

    #sotring the intial scalar_flux
    scalar_flux[:,0] = utl.ScalarFlux(inital_angular_flux, weights)
    
    source_mesh = np.ones([N_angles, N_ans], data_type)
    for i in range(N_mesh):
        for j in range(N_angles):
            source_mesh[j,2*i] = source[i]
            source_mesh[j,2*i+1] = source[i]
    
    angular_flux_last = inital_angular_flux
    
    for t in range(N_time):

        xsec_mesh_t = xsec_mesh + (1/(velocity* dt))
        source_mesh_tilde = source_mesh + angular_flux_last/(velocity* dt)
        
        
        start = timer()
        
        if (backend == 'OCI'):
            [angular_flux_total[:,:,t], current_total[:,t], spec_rad[t], source_converged, loops] = ss.OCI(sim_perams, dx_mesh, xsec_mesh_t, xsec_scatter_mesh, source_mesh_tilde, True)
        elif (backend == 'SI'):
            [angular_flux_total[:,:,t], current_total[:,t], spec_rad[t], source_converged, loops] = ss.SourceItteration(sim_perams, dx_mesh, xsec_mesh_t, xsec_scatter_mesh, source_mesh_tilde, True)
        else:
            print('>>>ERROR: NO Backend provided')
            print('     select between OCI and SI!')
            print()
            
        end = timer()
        
        
        if source_converged == False:
            print()
            print('>>>WARNING<<<')
            print('   Method of iteration did not converge!')
            print('')
            
        #angular_flux_total[:,:,t] = TimeDiscretization(angular_flux_last, angular_flux_half)
        
        #psi_check = source_mesh_tilde[0,5] / (xsec_mesh_t[5]*(1)/2)
        
        if (printer):
            print('Time step: {0}'.format(t))
            print('     -ρ:          {0}    '.format(spec_rad[t]))
            print('     -wall time:  {0} [s]'.format(end-start))
            print('     -loops:      {0}'.format(loops))
            print()
        #print('     -Ψ check:    {0}    '.format(psi_check))
        
        
        angular_flux_last = angular_flux_total[:,:,t]
        scalar_flux[:,t+1] = utl.ScalarFlux(angular_flux_last, weights)
        
    
    return(scalar_flux, current_total, spec_rad, loops)
    
    
    
    
#def TimeDiscretization(angular_flux_last, angular_flux_half, theta):
#    '''Using diamond discretization in time
#    '''
    
#    angular_flux_next = (angular_flux_half - theta*angular_flux_last) / (1-theta)
    
#    return(angular_flux_next)
