import numpy as np

def BoundaryCondition(BC, i, N_mesh, angular_flux=None, incident_flux_mag=None, angles=None, angle=None):
    
    
    if BC == 'reflecting':
        N_angles: int = angular_flux.shape[0]
        half = int(N_angles/2)
        psi_required = np.zeros(N_angles)
        
        if i == 0: #left edge
            psi_required[half:] = angular_flux[:half, 0]
            psi_required[:half] = angular_flux[half:, 0]
        else:
            psi_required[:half] = angular_flux[half:, -1]
            psi_required[half:] = angular_flux[:half, -1]
        
    elif BC == 'vacuum':
        psi_required = np.zeros(angular_flux.shape[0])
    
    elif BC == 'incident_iso':
        psi_required = BC_isotropic(incident_flux_mag, angles, i)
        
    elif BC == 'incident_ani':
        psi_required = BC_ani(incident_flux_mag, angle, angles)
        
    else:
        print()
        print('>>>Error: No Boundary Condition Supplied<<<')
        print()
    
    return(psi_required)

def BC_isotropic(incident_flux_mag, angles, i):

    BC = (incident_flux_mag)*np.ones(angles.size)
    half = int(angles.size/2)
    
    return(BC)

def BC_ani(incident_flux_mag, angle, angles):
    angle_id = find_nearest(angles, angle)
    BC = np.zeros(angles.size)
    BC[angle_id] = incident_flux_mag
    return(BC)
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (idx)
