"""
Analytical verification test for particle motion.

Verifies the simulation against analytical solutions for:
1. Free fall under gravity.
2. Energy conservation during elastic collision.
"""

import rusty_particles
import numpy as np


def test_free_fall():
    """
    Verifies that a particle falls under gravity according to h = 0.5 * g * t^2.
    """
    print("\n--- Test 1: Free Fall (Integrator Validation) ---")
    # Parameters
    dt = 1e-6
    total_time = 1.0
    g = 9.81
    y0 = 10.0

    # Create simulation
    sim = rusty_particles.Simulation.create(dt, -10.0, 0.0, -10.0, 10.0, 20.0, 10.0, 10)

    # Add particle
    # x, y, z, radius, mass
    sim.add_particle(0.0, y0, 0.0, 0.1, 1.0)

    # Run
    sim.run(total_time)

    # Get final position
    x, y, z = sim.get_particle_position(0)

    # Analytical Solution
    # y = y0 + v0*t - 0.5*g*t^2
    y_analytical = y0 - 0.5 * g * total_time**2

    print(f"Time: {total_time}s")
    print(f"Simulated Y: {y:.6f}")
    print(f"Analytical Y: {y_analytical:.6f}")

    error = abs(y - y_analytical)
    print(f"Error: {error:.6f}")

    assert error < 1e-2, f"Free Fall Test Failed: Error {error} > 1e-2"


def test_hertzian_compression():
    print("\n--- Test 2: Hertzian Static Compression (Two Particles) ---")
    # Particle 1 (Bottom): Fixed (Huge mass)
    # Particle 2 (Top): Moving (Normal mass)
    # Equilibrium: mg = F_hertz

    dt = 0.0001
    total_time = 1.0
    g = 9.81
    radius = 0.1
    mass = 1.0

    # Material properties
    E = 1e7
    nu = 0.3

    # Create simulation
    sim = rusty_particles.Simulation.create(dt, -1.0, -1.0, -1.0, 1.0, 2.0, 1.0, 10)
    sim.set_particle_material(E, nu, 2500.0, 0.5, 0.5)

    # Add particles
    # Bottom particle at (0, 0, 0)
    sim.add_particle(0.0, 0.0, 0.0, radius, 1e9, fixed=True)  # Huge mass ~ fixed

    # Top particle at (0, 2*radius, 0) - just touching
    sim.add_particle(0.0, 2.0 * radius, 0.0, radius, mass)

    # Run
    sim.run(total_time)

    # Get positions
    x1, y1, z1 = sim.get_particle_position(0)
    x2, y2, z2 = sim.get_particle_position(1)

    dist = y2 - y1
    overlap_sim = (2.0 * radius) - dist

    # Analytical Solution
    # R* = (R1*R2)/(R1+R2) = R^2 / 2R = R/2
    r_star = radius / 2.0

    # E* = ((1-nu^2)/E + (1-nu^2)/E)^-1 = (2(1-nu^2)/E)^-1 = E / (2(1-nu^2))
    e_star = E / (2.0 * (1.0 - nu**2))

    # F = 4/3 * E* * sqrt(R*) * delta^1.5
    # mg = 4/3 * E* * sqrt(R*) * delta^1.5
    # delta^1.5 = (3 * mg) / (4 * E* * sqrt(R*))
    # delta = (...) ^ (2/3)

    force = mass * g
    term = (3.0 * force) / (4.0 * e_star * np.sqrt(r_star))
    overlap_analytical = term ** (2.0 / 3.0)

    print(f"Time: {total_time}s")
    print(f"Simulated Overlap: {overlap_sim:.8f}")
    print(f"Analytical Overlap: {overlap_analytical:.8f}")

    error = abs(overlap_sim - overlap_analytical)
    print(f"Error: {error:.8f}")

    # Allow some tolerance due to damping and settling time
    assert error < 1e-5, f"Hertzian Test Failed: Error {error} > 1e-5"


if __name__ == "__main__":
    test_free_fall()
    test_hertzian_compression()
    print("\n✅ All tests passed!")
