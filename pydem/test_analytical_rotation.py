import rusty_particles
import pytest
import numpy as np


def test_ball_rolling_on_plate():
    # Parameters
    radius = 0.05
    mass = 1.0
    v0 = 1.0
    mu = 0.1
    g = 9.81
    dt = 0.0001

    # Analytical solution for time to saturation (pure rolling)
    # t_s = (2 * v0) / (7 * mu * g)
    expected_ts = (2 * v0) / (7 * mu * g)
    print(f"Expected time to saturation: {expected_ts:.4f}s")

    # Create simulation
    # Large bounds to avoid walls
    sim = rusty_particles.Simulation.create(dt, -10.0, 0.0, -10.0, 10.0, 10.0, 10.0, 1)

    # Set materials
    # High stiffness to minimize penetration effect on dynamics
    sim.set_particle_material(1e8, 0.3, 2500.0, mu, 0.5)
    sim.set_wall_material(1e8, 0.3, 2500.0, mu, 0.5)

    # Use Coulomb friction
    sim.set_contact_models("Hertzian", "Coulomb")

    # Add particle at some height (radius) so it touches the floor (y=0)
    # Floor is at y=0? No, bounds_min.y is 0.0.
    # But we need a floor mesh or just rely on bounds?
    # The simulation handles bounds as walls?
    # No, bounds are just for grid.
    # We need to add a floor mesh.

    # Create a floor mesh (two triangles)
    # Or just use a fixed particle as floor? No, flat plate.
    # Let's create a simple STL file or add mesh programmatically?
    # The Python API only has `add_mesh(path)`.
    # Let's create a temporary STL file.

    import struct

    def write_stl_triangle(f, v1, v2, v3):
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)
        f.write(struct.pack("<3f", *normal))
        f.write(struct.pack("<3f", *v1))
        f.write(struct.pack("<3f", *v2))
        f.write(struct.pack("<3f", *v3))
        f.write(struct.pack("<H", 0))

    stl_path = "floor.stl"
    with open(stl_path, "wb") as f:
        f.write(b"\x00" * 80)  # Header
        f.write(struct.pack("<I", 2))  # Number of triangles

        # Triangle 1
        v1 = np.array([-10.0, 0.0, -10.0])
        v2 = np.array([10.0, 0.0, -10.0])
        v3 = np.array([-10.0, 0.0, 10.0])
        write_stl_triangle(f, v1, v3, v2)  # Clockwise? Normal should point up (Y+)
        # v3-v1 = (0, 0, 20), v2-v1 = (20, 0, 0). Cross = (0, 400, 0). Up.

        # Triangle 2
        v1 = np.array([10.0, 0.0, -10.0])
        v2 = np.array([10.0, 0.0, 10.0])
        v3 = np.array([-10.0, 0.0, 10.0])
        write_stl_triangle(f, v1, v3, v2)

    sim.add_mesh(stl_path)

    # Add particle just touching the floor
    # y = radius - penetration? Ideally y=radius.
    sim.add_particle(0.0, radius, 0.0, radius, mass)

    # Set initial velocity
    sim.set_particle_velocity(0, v0, 0.0, 0.0)

    # Run simulation
    sim_time = 0.0
    saturated = False

    # Tolerance for floating point comparison
    tolerance = 0.05  # 5% error allowed

    for step in range(10000):  # Max steps
        sim.step()
        sim_time += dt

        # Check condition
        vx, vy, vz = sim.get_particle_velocity(0)
        wx, wy, wz = sim.get_particle_angular_velocity(0)

        # Velocity at contact point (bottom of sphere)
        # v_contact = v + w x r
        # r = (0, -radius, 0)
        # w = (0, 0, wz) (rolling around Z axis if moving in X)
        # v = (vx, 0, 0)
        # w x r = (wy*rz - wz*ry, wz*rx - wx*rz, wx*ry - wy*rx)
        #       = (0 - wz*(-R), 0, 0) = (wz*R, 0, 0)
        # v_contact_x = vx + wz * radius
        # No-slip condition: v_contact_x = 0 => vx = -wz * radius
        # Wait, if ball moves right (vx > 0), it rotates clockwise (wz < 0).
        # So vx = -wz * R is correct.
        # Or |vx| = |wz| * R.

        # Analytical derivation assumes v > 0 and w starts at 0.
        # Friction acts opposite to slip.
        # Slip velocity v_s = v - w*R (if w defined positive for rolling)
        # Here w is around Z.
        # v_contact = vx + wz * radius.
        # Initially vx=v0, wz=0 -> v_contact = v0 > 0.
        # Friction force f = -mu * N * sign(v_contact) = -mu * mg.
        # a = f/m = -mu * g.
        # Torque = r x f = (0, -R, 0) x (-f, 0, 0) = (0, 0, -R * f) ?
        # Wait, f is in -X direction.
        # r is (0, -R, 0).
        # r x f = (0, 0, (-R)*(-f) - 0) = (0, 0, R*f).
        # Torque is positive (counter-clockwise).
        # But if ball moves right, we expect clockwise rotation (negative Z).
        # Friction on ball is to the left (-X).
        # Torque = r x F. r=(0,-R,0). F=(-Ff, 0, 0).
        # r x F = (0, 0, 0 - (-R)*(-Ff)) = (0, 0, -R*Ff).
        # So Torque is negative (clockwise). Correct.
        # alpha = Torque / I = -R*mu*mg / (2/5 m R^2) = -5/2 * mu * g / R.

        # v(t) = v0 - mu*g*t
        # w(t) = 0 + alpha*t = -5/2 * mu * g / R * t
        # Contact velocity v_c = v + w*R = (v0 - mu*g*t) + (-5/2 * mu*g/R * t)*R
        # v_c = v0 - mu*g*t - 2.5 * mu*g*t = v0 - 3.5 * mu*g*t
        # Saturation when v_c = 0.
        # v0 = 3.5 * mu * g * t
        # t = v0 / (3.5 * mu * g) = 2*v0 / (7 * mu * g).
        # Correct.

        # Check if v_contact crosses zero
        v_contact = vx + wz * radius

        if v_contact <= 0.0:
            print(f"Saturation reached at t={sim_time:.4f}s")
            print(f"Velocity: {vx:.4f}, Angular Velocity: {wz:.4f}")
            print(f"v + wR: {vx + wz * radius:.4f}")
            saturated = True
            break

    if not saturated:
        pytest.fail("Did not reach saturation within max steps")

    # Assert
    error = abs(sim_time - expected_ts) / expected_ts
    print(f"Error: {error * 100:.2f}%")

    assert error < tolerance, (
        f"Simulation time {sim_time} differs from analytical {expected_ts} by {error * 100:.2f}%"
    )

    import os

    os.remove(stl_path)


if __name__ == "__main__":
    test_ball_rolling_on_plate()
