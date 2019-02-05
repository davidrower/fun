# Coulombic N-Body Simulation on Sphere

This is an idea I wanted to work on for a long time, and I finally realized it during my sophomore winter break. It is a simple simulation of charged particles on the surface of a sphere. The idea is to empirically solve for those stable and meta-stable configurations of N particles on a sphere, which are provoking and entertaining to look at!

## Some Details 

I use a Verlet style integrator to update particle positions and momenta. I have also updated a rather bootstrap-y annealing force in order to help the particles find more energertically favorable configurations.
