# Python API Reference

This document provides a detailed reference for the `rusty_particles` Python API.

## Module: `rusty_particles`

### Functions

#### `set_num_threads(num_threads: int)`
Sets the number of threads used by the simulation (via Rayon).
*   **num_threads**: Number of threads to use.

---

### Class: `Simulation`

The core simulation engine.

#### Static Methods

##### `create(dt, min_x, min_y, min_z, max_x, max_y, max_z, particle_count)`
Creates a new simulation instance.

*   **dt** *(float)*: Time step in seconds.
*   **min_x, min_y, min_z** *(float)*: Minimum coordinates of the simulation domain.
*   **max_x, max_y, max_z** *(float)*: Maximum coordinates of the simulation domain.
*   **particle_count** *(int)*: Initial capacity for particles (optimization hint).
*   **Returns**: A new `Simulation` instance.

#### Methods

##### `add_particle(x, y, z, radius, mass, fixed=False)`
Adds a spherical particle to the simulation.

*   **x, y, z** *(float)*: Position of the particle center.
*   **radius** *(float)*: Radius of the particle.
*   **mass** *(float)*: Mass of the particle.
*   **fixed** *(bool, optional)*: If `True`, the particle is immovable (default: `False`).

##### `add_mesh(path)`
Adds a static triangular mesh from an STL file.

*   **path** *(str)*: Path to the STL file.

##### `set_particle_material(youngs, poissons, density, friction, restitution, surface_energy=0.0)`
Sets the material properties for all particles.

*   **youngs** *(float)*: Young's Modulus (Pa).
*   **poissons** *(float)*: Poisson's Ratio.
*   **density** *(float)*: Density (kg/m³).
*   **friction** *(float)*: Coefficient of friction.
*   **restitution** *(float)*: Coefficient of restitution (0.0 to 1.0).
*   **surface_energy** *(float, optional)*: Surface energy in J/m² (default: 0.0). Used for JKR/sJKR models.

##### `set_wall_material(youngs, poissons, density, friction, restitution, surface_energy=0.0)`
Sets the material properties for walls.

*   **youngs** *(float)*: Young's Modulus (Pa).
*   **poissons** *(float)*: Poisson's Ratio.
*   **density** *(float)*: Density (kg/m³).
*   **friction** *(float)*: Coefficient of friction.
*   **restitution** *(float)*: Coefficient of restitution (0.0 to 1.0).
*   **surface_energy** *(float, optional)*: Surface energy in J/m² (default: 0.0).

##### `set_contact_models(normal_model, tangential_model)`
Configures the force models used for particle interactions.

*   **normal_model** *(str)*: The normal force model. Options:
    *   `"linear"` / `"spring_dashpot"`: Linear Spring-Dashpot.
    *   `"hertz"` / `"hertzian"`: Hertzian contact (non-linear).
    *   `"jkr"`: Johnson-Kendall-Roberts adhesion.
    *   `"sjkr"` / `"simplified_jkr"`: Simplified JKR (explicit adhesion).
*   **tangential_model** *(str)*: The tangential force model. Options:
    *   `"coulomb"`: Standard Coulomb friction.
    *   `"mindlin"`: Mindlin-Deresiewicz no-slip model.

##### `set_periodic_boundary(x, y, z)`
Enables or disables periodic boundary conditions for each axis.

*   **x, y, z** *(bool)*: `True` to enable periodicity for that axis.

##### `enable_gpu()`
Enables GPU acceleration. Requires a compatible GPU (Metal, Vulkan, or DX12).
*   **Raises**: `RuntimeError` if GPU initialization fails.

##### `step()`
Advances the simulation by one time step (`dt`).

##### `run(duration)`
Runs the simulation for a specified duration, showing a progress bar.

*   **duration** *(float)*: Total simulation time in seconds.

##### `get_particle_position(id)`
Retrieves the position of a particle.

*   **id** *(int)*: Particle ID (index).
*   **Returns**: `(x, y, z)` tuple.

##### `get_particle_velocity(id)`
Retrieves the velocity of a particle.

*   **id** *(int)*: Particle ID (index).
*   **Returns**: `(vx, vy, vz)` tuple.

##### `set_particle_velocity(id, vx, vy, vz)`
Sets the velocity of a particle.

*   **id** *(int)*: Particle ID (index).
*   **vx, vy, vz** *(float)*: New velocity components.

##### `get_particle_count()`
Returns the total number of particles.

*   **Returns**: *(int)* Count.

##### `write_vtk(filename)`
Writes the current simulation state to a VTK file (for visualization in ParaView).

*   **filename** *(str)*: Output file path (e.g., `"output.vtk"`).
