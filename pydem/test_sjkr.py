import rusty_particles
import numpy as np


def test_sjkr_adhesion():
    # Parameters
    radius = 0.005  # 5mm
    mass = 1.0  # Arbitrary
    gamma = 0.5  # Higher surface energy
    youngs = 1e5  # Softer material
    poissons = 0.3

    # Expected pull-off force for sJKR
    # F_pull = 1.5 * pi * gamma * R*
    r_star = radius / 2.0
    expected_pull_off = 1.5 * np.pi * gamma * r_star
    # Note: The force is negative (tensile), so we check magnitude.
    print(f"Expected Pull-off Force Magnitude: {expected_pull_off:.6f}")

    # Critical detachment distance
    # delta_c = -sqrt(3 * pi^2 * gamma^2 * R* / E*^2)
    inv_e_star = 2.0 * (1.0 - poissons**2) / youngs
    e_star = 1.0 / inv_e_star

    term = 3.0 * np.pi**2 * gamma**2 * r_star / e_star**2
    delta_c = -np.sqrt(term)
    print(f"Critical Detachment Distance: {delta_c:.6e}")

    # Create simulation
    dt = 1e-5
    sim = rusty_particles.Simulation.create(dt, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2)

    # Set materials
    sim.set_particle_material(youngs, poissons, 2500.0, 0.5, 0.5, gamma)

    # Set sJKR model
    sim.set_contact_models("sJKR", "Coulomb")

    # Add two particles
    sim.add_particle(0.0, 0.0, 0.0, radius, mass, True)

    # Start with small positive overlap (Loading)
    initial_overlap = radius * 0.01
    sim.add_particle(2.0 * radius - initial_overlap, 0.0, 0.0, radius, mass, False)

    # Step 1: Push together (Loading)
    # Force should be Hertzian.
    # F_Hertz = 4/3 * E* * sqrt(R*) * delta^1.5
    sim.step()  # Just one step to update force

    # We can't easily get force directly, but we can infer from acceleration.
    vx_prev, _, _ = sim.get_particle_velocity(1)
    sim.step()
    vx_curr, _, _ = sim.get_particle_velocity(1)
    ax = (vx_curr - vx_prev) / dt
    force_loading = mass * ax

    # Calculate expected Hertzian force
    x, _, _ = sim.get_particle_position(1)
    overlap = 2.0 * radius - x
    expected_hertz = (4.0 / 3.0) * e_star * np.sqrt(r_star) * overlap**1.5

    print(f"Loading Force: {force_loading:.6f}, Expected Hertz: {expected_hertz:.6f}")

    # Step 2: Pull apart (Unloading)
    # Set velocity to separate
    pull_vel = 0.01  # Slower pull
    sim.set_particle_velocity(1, pull_vel, 0.0, 0.0)

    min_force = 0.0

    # Run simulation to pull off
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

        # Check if separated beyond delta_c
        if overlap < delta_c * 1.1:  # Go slightly beyond
            break

    print(f"Minimum Force (Pull-off): {min_force:.6f}")

    # Check pull-off magnitude
    error = abs(abs(min_force) - expected_pull_off) / expected_pull_off
    print(f"Error: {error * 100:.2f}%")

    assert error < 0.1, (
        f"Pull-off force {min_force} differs from expected {expected_pull_off} by {error * 100:.2f}%"
    )


if __name__ == "__main__":
    test_sjkr_adhesion()
