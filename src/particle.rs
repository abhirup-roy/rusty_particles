use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub id: usize,
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub radius: f32,
    pub mass: f32,
}

impl Particle {
    pub fn new(id: usize, position: Vec3, radius: f32, mass: f32) -> Self {
        Self {
            id,
            position,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            radius,
            mass,
        }
    }

    pub fn update(&mut self, _dt: f32) {
        // Velocity Verlet (first half) is typically done in the solver, 
        // but simple Euler would be:
        // self.velocity += self.acceleration * dt;
        // self.position += self.velocity * dt;
        // self.acceleration = Vec3::ZERO; // Reset for next step
    }
}
