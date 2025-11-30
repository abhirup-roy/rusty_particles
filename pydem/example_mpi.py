import rusty_particles
import time
import sys


def main():
    print("Starting example_mpi.py...", flush=True)
    # Initialize simulation
    # Large domain to split
    sim = rusty_particles.Simulation.create(0.001, -2.0, 0.0, -1.0, 2.0, 5.0, 1.0, 0)

    # Initialize MPI
    try:
        sim.init_mpi()
    except Exception as e:
        print(f"MPI Init failed: {e}")
        return

    # Add particles (only on rank 0, or distributed?)
    # If we add on rank 0, they will be outside bounds for others?
    # Better: Each rank adds particles in its domain.
    # But for simplicity, let's add everywhere and let exchange_particles sort it out?
    # No, exchange_particles removes out of bounds.
    # So we should add particles that belong to us.

    # But we don't know our bounds easily in Python without exposing them.
    # Let's just add particles randomly and let them be removed if out of bounds?
    # Wait, `exchange_particles` logic:
    # if p.x < min_x -> send left
    # if p.x >= max_x -> send right
    # It assumes particles are initially valid or close to valid.

    # Let's add particles in a range and see.
    for i in range(100):
        x = -1.5 + i * 0.03  # -1.5 to 1.5
        sim.add_particle(x, 2.0, 0.0, 0.05, 1.0)

    # Run
    print("Running MPI Simulation...")
    sim.run(0.1)
    print("Done.")


if __name__ == "__main__":
    main()
