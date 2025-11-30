"""
Verification test for contact model selection.

Verifies that the simulation correctly accepts valid contact model names
and rejects invalid ones.
"""

import rusty_particles
import pytest


def test_contact_models():
    """
    Tests setting various contact models (Hertzian, JKR, sJKR, etc.).
    """
    print("Testing Configurable Contact Models...")

    # Create simulation
    sim = rusty_particles.Simulation.create(0.001, -1.0, 0.0, -1.0, 1.0, 5.0, 1.0, 100)

    # Test Default (Hertzian, Mindlin)
    print("Running with Default Models...")
    sim.run(0.1)

    # Test Linear Spring-Dashpot + Coulomb
    print("Setting models to Linear + Coulomb...")
    sim.set_contact_models("linear", "coulomb")
    sim.run(0.1)

    # Test Invalid Model
    print("Testing Invalid Model...")
    with pytest.raises(ValueError):
        sim.set_contact_models("invalid", "coulomb")


if __name__ == "__main__":
    test_contact_models()
