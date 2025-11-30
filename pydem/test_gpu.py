import rusty_particles
import pytest
import time


def test_gpu():
    print("Initializing simulation...")
    # dt, min_x, min_y, min_z, max_x, max_y, max_z, count
    sim = rusty_particles.Simulation.create(0.01, -1.0, 0.0, -1.0, 1.0, 5.0, 1.0, 1000)

    # Add particles
    print("Adding particles...")
    for i in range(1000):
        sim.add_particle(0.0, 2.0 + i * 0.01, 0.0, 0.05, 1.0)

    print("Enabling GPU...")
    try:
        sim.enable_gpu()
        print("GPU enabled successfully.")
    except Exception as e:
        pytest.skip(f"Skipping GPU test: {e}")

    print("Running GPU simulation...")
    start_time = time.time()
    for i in range(100):
        sim.step()
    end_time = time.time()

    print(f"Simulation complete in {end_time - start_time:.4f} seconds.")


if __name__ == "__main__":
    test_gpu()
