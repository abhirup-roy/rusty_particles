import rusty_particles
import numpy as np


def test_jkr_adhesion():
    # Parameters
    radius = 0.005  # 5mm
    mass = 1.0  # Arbitrary
    gamma = 0.1  # Surface energy
    youngs = 1e7
    poissons = 0.3

    # Expected pull-off force (User Limit)
    # F_pull = -2/3 * pi * gamma * R*
    r_star = radius / 2.0
    expected_pull_off = -(2.0 / 3.0) * np.pi * gamma * r_star
    print(f"Expected Pull-off Force: {expected_pull_off:.6f}")

    # Calculate Equilibrium Overlap (F=0)
    # a_eq^3 = 9 * pi * gamma * R*^2 / (2 * E*)
    # 1/E* = (1-nu^2)/E + (1-nu^2)/E = 2(1-nu^2)/E
    inv_e_star = 2.0 * (1.0 - poissons**2) / youngs
    e_star = 1.0 / inv_e_star

    a_eq = (9.0 * np.pi * gamma * r_star**2 / (2.0 * e_star)) ** (1.0 / 3.0)
    delta_eq = a_eq**2 / r_star - np.sqrt(8.0 * np.pi * gamma * a_eq / e_star)

    print(f"Equilibrium Overlap: {delta_eq:.6e}")

    # Create simulation with smaller dt
    dt = 1e-5
    sim = rusty_particles.Simulation.create(dt, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2)

    # Set materials with surface energy
    sim.set_particle_material(youngs, poissons, 2500.0, 0.5, 0.5, gamma)

    # Set JKR model
    sim.set_contact_models("JKR", "Coulomb")

    # Add two particles
    # Particle 0 at (0,0,0) fixed
    sim.add_particle(0.0, 0.0, 0.0, radius, mass, True)

    # Particle 1 at (2*radius - overlap, 0, 0)
    # Start at equilibrium overlap
    initial_overlap = delta_eq
    sim.add_particle(2.0 * radius - initial_overlap, 0.0, 0.0, radius, mass, False)

    # Pull apart slowly
    pull_vel = 0.002
    sim.set_particle_velocity(1, pull_vel, 0.0, 0.0)

    min_force = 0.0

    # Run simulation
    for i in range(50000):
        vx_prev, _, _ = sim.get_particle_velocity(1)
        sim.step()
        vx_curr, _, _ = sim.get_particle_velocity(1)

        ax = (vx_curr - vx_prev) / dt
        force = mass * ax

        if force < min_force:
            min_force = force

        x, _, _ = sim.get_particle_position(1)
        overlap = 2.0 * radius - x

        if overlap < -radius:
            break

    print(f"Minimum Force (Pull-off): {min_force:.6f}")

    error = abs(min_force - expected_pull_off) / abs(expected_pull_off)
    print(f"Error: {error * 100:.2f}%")

    # Allow 10% error (discrete steps)
    assert error < 0.1, (
        f"Pull-off force {min_force} differs from expected {expected_pull_off} by {error * 100:.2f}%"
    )


if __name__ == "__main__":
    test_jkr_adhesion()
