#!/usr/bin/python
"""
Purpose: Simulate charged particles confined to the surface of a sphere
         to find minimum energy configurations. The code does not account 
         for local vs global minima, so some interesting configurations 
         can be observed. 
         
Author:  David Rower
Date:    Sun Dec 25 04:29:27 2016
"""

print(__doc__)

import numpy as np
from matplotlib import pyplot as py
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def createParticles(N):
    """ Creates the arrays used to keep track of N particles. """
    positions  = 2 * np.random.rand(N,3) - 1.
    velocities = np.zeros((N,3))
    forces     = np.zeros((N,3))
    annealForces = np.zeros((N,3))
    return positions, velocities, forces, annealForces

def allForces(positions): 
    """ Creates an array of shape (N,3) of forces felt by each particle. """ 
    forces = np.zeros((N,3))
    for i in range(N-1): 
        for j in range(i+1,N):
            f = force(i,j,positions)
            forces[i] += f
            forces[j] -= f
    return forces

def force(i,j,positions): 
    """ Calculates force between two particles (Newton's 3rd law applies). """
    r1,r2 = positions[i],positions[j]
    rVec  = r1 - r2
    r     = np.linalg.norm(rVec)
    fVec  = rVec / r**3
    return fVec

def velocityVerlet(positions, velocities, forces):
    """ Updates state according to the velocity Verlet algorithm. """
    velocities += forces * .5 * dt                  # velocities to n+.5
    positions  += velocities * dt                   # positions to n+1
    norms       = np.linalg.norm(positions,axis=1)  # confines particles to 
    positions   = positions / norms[:,np.newaxis]     # surface of unit sphere
    forces      = allForces(positions)              # forces re-calculated
    velocities += forces * 0.5 * dt                 # velocities to n+1
    return positions, velocities, forces

def animate(i):
    """ Function to update the data to be plotted in frame i. """
    global positions, velocities, forces, annealForces
    positions, velocities, forces = velocityVerlet(positions,velocities,forces)
    annealForces = np.random.randn(len(positions),3)
    norms   = np.linalg.norm(annealForces,axis=1)         
    forces += 10./(i+1) * annealForces/norms[:,np.newaxis] 
    for index, value in enumerate(positions):
        x, y, z = positions[index][0], positions[index][1], positions[index][2]
        pts[index].set_data(x,y)
        pts[index].set_3d_properties(z)
    return pts

def init(): 
    for pt in pts:
        pt.set_data([], [])
        pt.set_3d_properties([])
    return pts

def createAnimation(N, dt): 
    global pts, positions, velocities, forces, annealForces
    
    # create our figure and 3d axis
    fig = py.figure()
    ax  = p3.Axes3D(fig, aspect='equal', autoscale_on=False)
    
    # set axis properties    
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')    
   
    # create the arrays to keep track of our particles
    positions, velocities, forces, annealForces = createParticles(N)

    # pts is modified in the animation object
    pts = sum([ax.plot([], [], [], 'o', markersize=8, color='k')
           for i in range(N)], [])

    # Creating the Animation object
    ani = animation.FuncAnimation(fig, animate, init_func = init,
                                                interval=.1, blit=True)
                                  
    # Create and plot a unit sphere
    phi, theta = np.mgrid[0.0:np.pi:30j, 0.0:2.0 * np.pi:30j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)
    
    py.show()

if __name__ == "__main__": 
    N  = int(input("Number of Particles? "))
    dt = 0.01
    createAnimation(N, dt)

    
        
    
