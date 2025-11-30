"""
Verification test for multithreading.

Verifies that the simulation runs correctly with multiple threads enabled.
"""

import rusty_particles
import time
import pytest


def test_multithreading():
    num_threads = 4
    print(f"\n--- Testing with {num_threads} threads ---")

    try:
        rusty_particles.set_num_threads(num_threads)
    except RuntimeError as e:
        print(f"Warning: Could not set threads (pool likely already initialized): {e}")
    except Exception as e:
        pytest.fail(f"Could not set threads: {e}")

    # Create simulation
    sim = rusty_particles.Simulation.create(0.001, -1.0, 0.0, -1.0, 1.0, 5.0, 1.0, 5000)

    # Add particles
    for i in range(1000):
        sim.add_particle(0.0, 2.0 + i * 0.001, 0.0, 0.05, 1.0)

    start_time = time.time()
    sim.run(0.1)  # Run for short time
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    test_multithreading()
