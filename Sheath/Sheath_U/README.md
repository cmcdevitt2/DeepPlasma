# Fluid PINN

## Project Description
This repository contains a hierarchy of Physics-Informed Neural Networks (PINN) designed to solve increasingly complex steady-state fluid equations for a plasma near a material wall, i.e. the plasma sheath. 

The PINNs employ specialized input/output transforms to enforce physical symmetries and boundary conditions and a 2nd order quasi-newton optimizer that allows for greater convergence.

<p float="left">
 <img src="Figures_readme/Velocities.png" width="24%" />
 <img src="Figures_readme/phi.png" width="24%" />
 <img src="Figures_readme/q.png" width="24%" />
 <img src="Figures_readme/Losses.png" width="24%" />
</p>

> **Figure 1:** Velocity profiles from the first model, potential profiles from the second model, and heat transfer profile from the third model with an example training history from the first model. 

## Prerequisites
Prerequisites vary per model and are described in detail within each model directory.

## Setup and Execution
Setup and execution varies per model and is described in detail within each model directory.

## Reference
[1] Webb, Ethan, Yuzhi Li, and Christopher J. McDevitt. "A Deep Learning Approach to Describing the Plasma Sheath." *Plasma Sources Science and Technology* (2026).
