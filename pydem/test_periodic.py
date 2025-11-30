import random
import rusty_particles
import pytest


def test_periodic():
    # Create simulation with small bounds
    # dt, min_x, min_y, min_z, max_x, max_y, max_z, count
    sim = rusty_particles.Simulation.create(0.01, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1)

    # Enable periodic boundaries in X
    sim.set_periodic(False, True, False)

    # Add a particle moving towards +X boundary
    # x, y, z, radius, mass
    sim.add_particle(0.9, 0.0, 0.0, 0.1, 1.0)

    # Let's test Y wrapping with gravity.
    sim.set_periodic(False, True, False)

    # Add particle at top
    particle_count = 100
    for i in range(particle_count):
        x = random.uniform(-0.4, 0.4)
        z = random.uniform(-0.4, 0.4)
        y = random.uniform(1.0, 4.0)
        radius = random.uniform(0.02, 0.03)
        mass = 1.0
        sim.add_particle(x, y, z, radius, mass)

    print("Running periodic simulation...")

    # Run for a few steps to ensure no crashes
    try:
        for i in range(100):
            sim.step()
    except Exception as e:
        pytest.fail(f"Simulation crashed: {e}")


if __name__ == "__main__":
    test_periodic()
