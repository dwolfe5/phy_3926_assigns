# Members: Dakota, Shane, Robbie, and Ash

# PART 1

import numpy as np
import matplotlib.pyplot as plt

def diffusion_ftcs(nspace, ntime, tau_rel, args):

    """
    Solve the 1D diffusion equation using the Forward Time Centered Space (FTCS) scheme.

    Parameters:
    -----------------------------------------------------------------------------------
    nspace (int): Number of spatial grid points.

    ntime (int): Number of time steps.

    tau_rel (float): Time step relative to the critical time step.

    args (list): List of problem parameters - [bar length (L), thermal diffusion coefficient (kappa)].

    Returns:
    ------------------------------------------------------------------------------------
    tuple: Tuple containing three arrays - xplot, tplot, ttplot. 

           xplot: Array of spatial positions.

           tplot: Array of time values.

           ttplot: 2D array containing the temperature field T(x, t).

    """

    # Assign parameters from the args list
    L, kappa = args
    
    # Initialize parameters
    h = L / (nspace - 1)
    tau = tau_rel * (h ** 2) / (2 * kappa) # using equation 6.28
    coeff = kappa * tau / (h ** 2)
    
    if coeff < 0.5:
        print('Solution is expected to be stable')
    else:
        print('WARNING: Solution is expected to be unstable')
    
    # Set initial and boundary conditions
    tt = np.zeros(nspace)                 # boundary
    tt[nspace // 2] = 1.0 / h             # intital condition (delta function)
    
    # Initialize arrays for plotting
    xplot = np.linspace(-L/2, L/2, nspace)
    ttplot = np.empty((nspace, ntime))
    tplot = np.empty(ntime)

    iplot = 0
    nplots = 50
    plot_step = ntime // nplots
    
    # Time-stepping loop
    for istep in range(ntime):
        tt[1:(nspace-1)] = tt[1:(nspace-1)] + coeff * (tt[2:nspace] + tt[0:(nspace-2)] - 2 * tt[1:(nspace-1)])
        
        # Record temperature for plotting
        ttplot[:, istep] = np.copy(tt)
        tplot[istep] = (istep + 1) * tau

        if (istep + 1) % plot_step == 0:
            ttplot[:, iplot] = np.copy(tt)
            tplot[iplot] = (istep + 1) * tau
            iplot += 1
    
    return xplot, tplot, ttplot

# Define problem parameters
tau_rel = 0.4                             # Time step relative to the critical time step
args = [1.0, 1.0]                         # Bar length (L) and thermal diffusion coefficient (kappa)
ntime = 100                               # The number of time steps  
nspace = 101                              # The number of grid points 

# Call this function using the problem parameters
xplot, tplot, ttplot = diffusion_ftcs(nspace, ntime, tau_rel, args)

print(ttplot)                             # check that T(x,y) is a 2D array

# Plot the results as a contour plot
levels = np.linspace(0., 10., num=21)
ct = plt.contour(tplot, xplot, ttplot, levels)
plt.clabel(ct, fmt='%1.2f')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Temperature contour plot')
plt.show()

# PART 2

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Set parameters for reproducing Figure 6.7
tau7 = 1e-4 # Given in figure
args7 = [1.0, 1.0]  
ntime7 = int(0.03 // tau7)          
nspace7 = 61 # Given in figure    
h7 = args7[0] / ( nspace7 - 1 )            
tau_rel7 = tau7 * ( 2 * args7[1] ) / (h7 ** 2)                

xplot7, tplot7, ttplot7 = diffusion_ftcs(nspace7, ntime7, tau_rel7, args7)

# print(tau_rel7)
# print(tau7)

# Plot the results as a contour plot
fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')
Tp, Xp = np.meshgrid(tplot7, xplot7)
ax.plot_surface(Tp, Xp, ttplot7, rstride=2, cstride=2, cmap=cm.gray)
ax.set_xlabel('Time')
ax.set_ylabel('x')
ax.set_zlabel('T(x,t)')
ax.set_title('Diffusion of a delta spike')

fig3 = plt.figure()
levels7 = np.linspace(0., 10., num=21)
ct7 = plt.contour(tplot7, xplot7, ttplot7, levels7)
plt.clabel(ct, fmt='%1.2f')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Temperature contour plot')
plt.show()
