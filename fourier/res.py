import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math

p2 = np.pi*2

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

N_angle = 8

def complex2cont(comp, x):
    y = comp.real * np.cos(comp.imag * p2 * x) + comp.real * np.sin( comp.imag * p2 * x)
    return(y)

dom_eig = 0.4299660597338153-0.220210347740101j

print(np.abs(dom_eig))

data = np.genfromtxt('res.csv', dtype=np.float64, delimiter=',', skip_header=2) #delimiter=','

res = data[:,0]
spec_rad = data[:,1]

x = np.linspace(0,res.size, res.size)

spec_rad_pred = np.abs(dom_eig)*np.ones(res.size)

ana = complex2cont(dom_eig, x)

ana *= .1 
ana += np.abs(dom_eig)
#ana *= -1

plt.figure(1)
plt.plot(x, spec_rad, '-x', label='Empirical', color='#990223', linewidth=2)
plt.plot(x, spec_rad_pred, '--k', label=r'$|λ_{max}|$', linewidth=2)
#plt.plot(x, ana, label=r'$0.1(λ_r cis(2πλ_ix)) + |λ|$')
plt.xlabel("x [iteration]")
plt.ylabel("ρ")
plt.legend()
plt.ylim((.35,.65))
plt.grid()
#plt.title("Spectral Radius") #"λ={}".format(dom_eig)
plt.savefig('eig_spec_rad.pdf')



plt.figure(2)
plt.plot(x, res, 'k-', linewidth=2)
plt.yscale('log')
plt.ylabel(r'residual $|ψ-0|_2$')
plt.xlabel('x [iteration]')
plt.grid()
plt.savefig('eig_res.pdf')