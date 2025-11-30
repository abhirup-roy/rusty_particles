# rusty-particles

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![Rust](https://img.shields.io/badge/rust-2021-orange)

**rusty-particles** is a high-performance, parallel Discrete Element Method (DEM) simulation engine written in Rust with Python bindings. It is designed for simulating granular materials with efficiency and flexibility, leveraging modern hardware capabilities including multi-core CPUs and GPUs.

## 🚀 Features

- **Hybrid Compute Engine**:
  - **CPU**: Highly optimized parallel execution using `rayon` (multithreading) and `mpi` (distributed computing).
  - **GPU**: Massively parallel simulation using `wgpu` compute shaders for extreme performance.
- **Python Interface**: Full control over simulation setup, execution, and analysis via a user-friendly Python API (`pyo3`).
- **Advanced Physics**:
  - Configurable **Normal Force Models**: Hertzian, Linear, Hysteretic.
  - Configurable **Tangential Force Models**: Mindlin, Coulomb, Linear.
  - Periodic Boundary Conditions.
- **Visualization**: Built-in support for exporting simulation data to **VTK** format for visualization in ParaView or Ovito.
- **Cross-Platform**: Runs on macOS (Apple Silicon/Intel), Linux, and Windows.

## 🛠️ Installation

### Prerequisites
- **Rust**: Install via [rustup](https://rustup.rs/).
- **Python**: Version 3.7 or newer.
- **MPI** (Optional, for distributed runs): OpenMPI or MPICH.
  - *macOS*: `brew install open-mpi`
  - *Ubuntu*: `sudo apt install libopenmpi-dev`

### Install via Pip
You can install the package directly from the source directory:

```bash
pip install .
```

This will compile the Rust backend and install the Python package.

### Development Build
For active development, use `maturin` to build and install in editable mode:

```bash
pip install maturin
maturin develop --release
```

## ⚡ Quick Start

Here is a simple example of a simulation dropping particles into a box:

```python
import rusty_particles

# Create a simulation
# (dt, min_x, min_y, min_z, max_x, max_y, max_z, capacity)
sim = rusty_particles.Simulation.create(0.001, -1.0, 0.0, -1.0, 1.0, 5.0, 1.0, 10000)

# Add particles
for i in range(100):
    sim.add_particle(0.0, 2.0 + i * 0.1, 0.0, 0.05, 1.0)

# Run simulation
print("Running simulation...")
sim.run(1.0) # Run for 1.0 second
print("Done!")
```

## 🏎️ Advanced Usage

### GPU Acceleration
Enable GPU mode to offload physics calculations to the graphics card:

```python
sim.enable_gpu()
sim.run(1.0)
```

### Multithreading (CPU)
Control the number of threads used by the CPU engine (useful for macOS/Single-node):

```python
rusty_particles.set_num_threads(8)
sim.run(1.0)
```

### MPI (Distributed Computing)
For large-scale simulations across multiple nodes (Linux recommended):

```python
# Initialize MPI environment
sim.init_mpi()
sim.run(1.0)
```
*Note: Run with `mpirun -n <N> python script.py`.*

### Custom Physics Models
Select specific contact models for your material:

```python
sim.set_contact_models("linear", "coulomb")
# Options:
# Normal: "hertz" (default), "linear", "hysteretic"
# Tangential: "mindlin" (default), "coulomb", "linear"
```

For detailed information on the physics models, see [Physics Documentation](docs/PHYSICS.md).

## 📂 Project Structure

- `src/`: Rust source code (core engine).
  - `simulation.rs`: Main simulation loop and logic.
  - `particle.rs`: Particle data structure.
  - `physics.rs`: Contact force models.
  - `gpu.rs`: WGPU integration.
- `pydem/`: Python examples and test scripts.
- `src/shaders/`: WGSL shaders for GPU compute.

For a high-level overview of the system design, see [Architecture Documentation](docs/ARCHITECTURE.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
