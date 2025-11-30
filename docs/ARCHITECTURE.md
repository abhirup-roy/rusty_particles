# Architecture

`rusty-particles` is designed as a high-performance Rust core with a user-friendly Python interface.

## High-Level Overview

```mermaid
graph TD
    Python[Python API (pyo3)] --> Rust[Rust Core]
    Rust --> Simulation[Simulation Loop]
    Simulation --> CPU[CPU Engine]
    Simulation --> GPU[GPU Engine (wgpu)]
    
    CPU --> Rayon[Rayon (Multithreading)]
    CPU --> MPI[MPI (Distributed)]
    
    GPU --> Compute[Compute Shaders (WGSL)]
```

## Core Components

### 1. Simulation (`src/simulation.rs`)
The central coordinator. It holds the state of the world (particles, grid, boundaries) and manages the simulation loop. It decides whether to step the simulation on the CPU or GPU.

### 2. Particle System (`src/particle.rs`)
Defines the `Particle` struct. Data is stored in a Structure-of-Arrays (SoA) layout on the GPU for performance, but as a Vector of Structs (AoS) on the CPU for flexibility.

### 3. Grid (`src/grid.rs`)
Implements a spatial hashing grid (Uniform Grid) for broad-phase collision detection. This reduces the complexity of finding neighbors from $O(N^2)$ to $O(N)$ on average.

### 4. Physics Engine (`src/physics.rs`)
Contains the implementation of the contact force models. It is stateless and operates on pairs of particles identified by the grid.

### 5. GPU Backend (`src/gpu.rs`)
Manages the `wgpu` context, buffers, and compute pipelines. It synchronizes data between the CPU and GPU only when necessary (e.g., for initialization or visualization output).

## Python Bindings (`src/lib.rs`)
Uses `pyo3` to expose Rust structs and functions as Python classes and modules.
- `Simulation` class: Exposes methods like `add_particle`, `run`, `enable_gpu`.
- `set_num_threads`: Exposes `rayon` configuration.

## Parallelism Strategies

### Multithreading (Shared Memory)
Uses `rayon` to parallelize the main simulation loop on the CPU.
- **Particle Integration**: Parallel iterator over particles.
- **Force Calculation**: Parallel iterator with a thread-local or atomic accumulation strategy (currently using a lock-free approach or DashMap for contacts).

### MPI (Distributed Memory)
Uses `mpi` crate for domain decomposition.
- **Decomposition**: 1D Slab decomposition along the X-axis.
- **Ghost Layer**: Particles near the boundary of a rank are sent to neighbors as "ghosts" to allow force calculation across ranks.
- **Migration**: Particles leaving a rank's domain are transferred to the owning rank.

### GPU (Massive Parallelism)
Uses Compute Shaders written in WGSL.
- **Workgroups**: Particles are processed in parallel workgroups.
- **Synchronization**: `storage` buffers are used to share particle data. Atomic operations are used where necessary.
