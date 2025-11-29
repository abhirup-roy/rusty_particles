import rusty_particles
import time
import os
import random


def main():
    print("Initializing simulation from Python...")

    # Parameters
    dt = 0.001
    particle_count = 1000

    # Create simulation using factory method
    # dt, min_x, min_y, min_z, max_x, max_y, max_z, count
    sim = rusty_particles.Simulation.create(
        dt, -1.0, 0.0, -1.0, 1.0, 5.0, 1.0, particle_count
    )

    # Add particles
    print(f"Adding {particle_count} particles...")
    for i in range(particle_count):
        x = random.uniform(-0.4, 0.4)
        z = random.uniform(-0.4, 0.4)
        y = random.uniform(1.0, 4.0)
        radius = random.uniform(0.02, 0.03)
        mass = 1.0
        sim.add_particle(x, y, z, radius, mass)

    # Run simulation
    steps = 500
    print(f"Running for {steps} steps...")

    if not os.path.exists("output_py"):
        os.makedirs("output_py")

    start_time = time.time()
    for step in range(steps):
        sim.step()

        if step % 10 == 0:
            sim.write_vtk(f"output_py/step_{step:05d}.vtk")

    end_time = time.time()
    print(f"Simulation complete in {end_time - start_time:.2f} seconds.")
    print("Output saved to output_py/")


if __name__ == "__main__":
    main()
