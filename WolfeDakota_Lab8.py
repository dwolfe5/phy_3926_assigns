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
