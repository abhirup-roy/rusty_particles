import rusty_particles

print("Testing simple function...")
try:
    print(rusty_particles.hello())
except Exception as e:
    print(f"Error: {e}")

print("Initializing simulation (factory method)...")
# dt, min_x, min_y, min_z, max_x, max_y, max_z, count
sim = rusty_particles.Simulation.create(0.001, -1.0, 0.0, -1.0, 1.0, 5.0, 1.0, 100)
print("Done.")
