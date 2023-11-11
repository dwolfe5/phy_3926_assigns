# Members: Dakota, Shane, Robert, and Ashton

# PART 1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
    
    if tau_rel < 1:
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
    
    # Time-stepping loop
    for istep in range(ntime):
        tt[1:(nspace-1)] = tt[1:(nspace-1)] + (tau_rel / 2.0) * (tt[2:nspace] + tt[0:(nspace-2)] - 2 * tt[1:(nspace-1)])
        
        # Record temperature for plotting
        # We do this at every time step since the problem does not specify to plot at a smaller time domain than what is calculated
        ttplot[:, istep] = np.copy(tt)
        tplot[istep] = (istep + 1) * tau_rel * (h**2)/(2*kappa)
    
    return xplot, tplot, ttplot

# PART 2

# Set parameters for reproducing Figure 6.7
tau_67 = 1e-4 # Given in figure
args = [1.0, 1.0]  
# Time in both plots of figure go up to ~0.03s
ntime_67 = int(0.03 // tau_67)          
nspace = 61 # Given in figure    
h = args[0] / ( nspace - 1 )            
tau_rel_67 = tau_67 * ( 2 * args[1] ) / (h ** 2)                

# Calculate values to plot
xplot_67, tplot_67, ttplot_67 = diffusion_ftcs(nspace, ntime_67, tau_rel_67, args)

fig_67 = plt.figure()

# Mesh plot
ax1 = fig_67.add_subplot(1,2,1, projection = '3d')
Tp, Xp = np.meshgrid(tplot_67, xplot_67)
ax1.plot_surface(Tp, Xp, ttplot_67, rstride=2, cstride=2, cmap=cm.gray)
ax1.set_xlabel('Time')
ax1.set_ylabel('x')
ax1.set_zlabel('T(x,t)')
ax1.set_title('Diffusion of a delta spike')

# Contour plot
ax2 = fig_67.add_subplot(1,2,2)
levels = np.linspace(0., 10., num=21)
ct_67 = ax2.contour(tplot_67, xplot_67, ttplot_67, levels)
ax2.clabel(ct_67, fmt='%1.2f')
ax2.set_xlabel('Time')
ax2.set_ylabel('x')
ax2.set_title('Temperature contour plot')
plt.tight_layout()
plt.show()

# Set parameters for reproducing Figure 6.8
tau_68 = 1.5e-4 # Given in figure
# Time in mesh plot of figure 6.8 goes up to ~0.06
ntime_68 = int(0.06 // tau_68)               
tau_rel_68 = tau_68 * ( 2 * args[1] ) / (h ** 2) 

# Calculate values to plot
xplot_68, tplot_68, ttplot_68 = diffusion_ftcs(nspace, ntime_68, tau_rel_68, args)

fig_68 = plt.figure() 

# Plot unstable meshgrid
ax3 = fig_68.add_subplot(1,2,1,projection='3d')
Tp2, Xp2 = np.meshgrid(tplot_68, xplot_68)
ax3.plot_surface(Tp2, Xp2, ttplot_68, rstride=2, cstride=2, cmap=cm.gray)
ax3.set_xlabel('Time')
ax3.set_ylabel('x')
ax3.set_zlabel('T(x,t)')
ax3.set_title('Diffusion of a delta spike')

# Plot unstable contour
ax4 = fig_68.add_subplot(1,2,2)
# For contour in 6.8, the time only goes up to ~0.045 so we cut off the arrays then
new_indices = tplot_68 <= 0.045
ct_68 = plt.contour(tplot_68[new_indices], xplot_68, ttplot_68[:,new_indices], levels)
plt.clabel(ct_68, fmt='%1.2f')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Temperature contour plot')
plt.show()
