## Physics Description

This repository contains two distinct solvers designed to evaluate the **mean escape time** of energetic ions in axisymmetric tokamak geometry. The framework solves the inhomogeneous adjoint of the drift kinetic equation, providing a metric for energetic particle transport due to both direct orbit loss and collisional transport [1].

### Codes Included

#### 1. Physics-Informed Neural Network (PINN)
The code solves the steady-state drift kinetic equation for ions subject to:

*   **Guiding Center Drifts:** Includes parallel streaming, $\vec{E} \times \vec{B}$ drift, and magnetic gradient/curvature drifts [1].
*   **Collisions:** A Lorentz collision operator is used to model pitch-angle scattering [1].

The mean escape time, $T_s$, is computed as the solution to the inhomogeneous adjoint equation:

$$ \dot{X} \cdot \nabla T_s + \dot{V}_{\parallel} \frac{\partial T_s}{\partial v_{\parallel}} + C^*_s(T_s) = -1 $$

Provides a surrogate model capable of predicting the mean escape time across the entire phase space [1].

<p float="left">
  <img src="figures/T_Z_R_xi0o3_Dec7_2025.png" width="24%" />
  <img src="figures/T_Z_R_xi0o8_Dec7_2025.png" width="24%" />
  <img src="figures/T_Z_R_xim0o3_Dec7_2025.png" width="24%" />
  <img src="figures/T_Z_R_xim0o8_Dec7_2025.png" width="24%" />
</p>

> **Figure 2:** Mean escape time (log scale) for ions with different initial pitches, showing phase space regions of good fast ion confinement and versus prompt loss [1].

#### 2. JONTA (Just anOther fuNcTionAl pusher) Ion Guiding Center Module
A GPU-accelerated particle-based solver built on JAX and PyTorch. It utilizes a Runge-Kutta integration scheme for the guiding center equations and a Monte Carlo operator for collisions [1].

<p float="left">
  <img src="figures/orbit_CoCurrent.png" width="33%" />
  <img src="figures/orbit_CounterCurrent.png" width="33%" /> 
</p>

> **Figure 1:** Example collisionless ion orbits in the circular flux surface geometry used by JONTA. **Left:** Co-current passing and trapped orbits. **Right:** Counter-current passing and trapped orbits [1].



## References
[1] C. J. McDevitt and J. S. Arnaud, "An Adjoint Formulation of Energetic Particle Confinement," Submitted to the Journal of Plasma Physics (Preprint: arXiv:2511.11968), 2026.