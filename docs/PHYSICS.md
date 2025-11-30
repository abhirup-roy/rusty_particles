# Physics Models

`rusty-particles` supports several contact force models to simulate different types of granular materials. These models calculate the forces between interacting particles based on their overlap, relative velocity, and material properties.

## Normal Force Models

The normal force ($F_n$) acts along the line connecting the centers of two colliding particles.

### 1. Hertzian (Default)
Based on Hertz contact theory, suitable for elastic spheres.
- **Formula**: $F_n = k_n \delta^{3/2} - \gamma_n v_n \sqrt{\delta}$
- **Parameters**:
  - $k_n$: Stiffness constant (derived from Young's Modulus and Poisson's Ratio).
  - $\delta$: Overlap distance.
  - $\gamma_n$: Damping coefficient.
  - $v_n$: Normal relative velocity.

### 2. Linear
A simple spring-dashpot model.
- **Formula**: $F_n = k_n \delta - \gamma_n v_n$
- **Use Case**: Simple simulations where computational speed is prioritized over physical accuracy for elastic deformations.

### 3. Hysteretic
Captures plastic deformation or energy loss during impact more accurately than simple damping.
- **Formula**: $F_n = k_{load} \delta$ (loading) / $k_{unload} (\delta - \delta_0)$ (unloading).
- **Use Case**: Cohesive or plastic materials.

## Tangential Force Models

The tangential force ($F_t$) acts perpendicular to the normal force, opposing the relative sliding motion (friction).

### 1. Mindlin (Default)
Based on Mindlin-Deresiewicz theory, accounting for micro-slip and history-dependent friction.
- **Features**: Accurate representation of static and dynamic friction for elastic spheres.
- **Limit**: Clamped by Coulomb friction limit ($\mu F_n$).

### 2. Coulomb
Standard Coulomb friction model.
- **Formula**: $F_t = -\mu F_n \cdot \text{sgn}(v_t)$
- **Use Case**: Simple sliding friction.

### 3. Linear
Linear spring-dashpot in the tangential direction.
- **Formula**: $F_t = -k_t \delta_t - \gamma_t v_t$
- **Limit**: Clamped by Coulomb friction limit.

## Material Properties

The behavior of these models is governed by the material properties assigned to particles:
- **Young's Modulus ($E$)**: Stiffness of the material.
- **Poisson's Ratio ($\nu$)**: Ratio of transverse contraction strain to longitudinal extension strain.
- **Density ($\rho$)**: Mass density.
- **Restitution Coefficient ($e$)**: Controls energy loss during collisions (0 = perfectly inelastic, 1 = perfectly elastic).
- **Friction Coefficient ($\mu$)**: Controls sliding resistance.
