# Rusty Particles

A high-performance Discrete Element Method (DEM) simulation engine written in Rust with Python bindings.

## Features

*   **High Performance:** Core engine written in Rust for maximum speed and memory efficiency.
*   **Python Bindings:** Easy-to-use Python API via `pyo3`.
*   **Contact Models:**
    *   Linear Spring-Dashpot
    *   Hertzian Contact (Non-linear elasticity)
    *   JKR (Johnson-Kendall-Roberts) Adhesion
    *   sJKR (Simplified JKR) - Explicit, stable approximation for adhesion
    *   Coulomb Friction
    *   Mindlin-Deresiewicz (No-slip)
*   **GPU Acceleration:** Optional GPU support using `wgpu` (Metal, Vulkan, DX12).
*   **Parallelism:** Multi-threaded execution via `rayon` and distributed computing via MPI.
*   **Complex Boundaries:** Support for triangular meshes (STL) as static boundaries.

## Installation

### Prerequisites

*   **Rust:** [Install Rust](https://www.rust-lang.org/tools/install)
*   **Python:** Python 3.8+
*   **Maturin:** `pip install maturin`

### Building from Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/rusty-particles.git
    cd rusty-particles
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install dependencies and build:
    ```bash
    pip install -r requirements.txt
    maturin develop --release
    ```

    To enable GPU support (requires compatible hardware):
    ```bash
    maturin develop --release --features extension-module
    ```

## Usage

For detailed API documentation, see [API_REFERENCE.md](API_REFERENCE.md).

### Basic Example

```python
import rusty_particles

# Create a simulation
# dt=1e-4, bounds=(-1,-1,-1) to (1,1,1), capacity=1000
sim = rusty_particles.Simulation.create(1e-4, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1000)

# Set material properties
# Young's Modulus, Poisson's Ratio, Density, Restitution, Friction, Surface Energy
sim.set_particle_material(1e7, 0.3, 2500.0, 0.5, 0.5, 0.0)

# Add particles
sim.add_particle(0.0, 0.5, 0.0, 0.05, 1.0) # x, y, z, radius, mass

# Run simulation
sim.run(1.0) # Run for 1.0 second
```

### Advanced Configuration

#### Contact Models

```python
# Use Hertzian contact with Coulomb friction
sim.set_contact_models("Hertzian", "Coulomb")

# Use Simplified JKR (sJKR) for adhesive particles
sim.set_contact_models("sJKR", "Coulomb")
```

#### GPU Acceleration

```python
try:
    sim.enable_gpu()
    print("GPU enabled!")
except Exception as e:
    print(f"GPU init failed: {e}")
```

## Testing

Run the verification suite using `pytest`:

```bash
pytest pydem/
```

## License

MIT License
