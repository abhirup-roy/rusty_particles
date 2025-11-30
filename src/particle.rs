use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub id: usize,
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub radius: f32,
    pub mass: f32,
    pub fixed: bool,
    // Kahan summation residuals
    pub position_residual: Vec3,
    pub velocity_residual: Vec3,
}

impl Particle {
    pub fn new(id: usize, position: Vec3, radius: f32, mass: f32, fixed: bool) -> Self {
        Self {
            id,
            position,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            radius,
            mass,
            fixed,
            position_residual: Vec3::ZERO,
            velocity_residual: Vec3::ZERO,
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

#[derive(Clone, Copy, Debug, mpi::traits::Equivalence)]
pub struct MpiParticle {
    pub id: usize,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub acceleration: [f32; 3],
    pub radius: f32,
    pub mass: f32,
    pub fixed: bool,
}

impl From<Particle> for MpiParticle {
    fn from(p: Particle) -> Self {
        Self {
            id: p.id,
            position: p.position.to_array(),
            velocity: p.velocity.to_array(),
            acceleration: p.acceleration.to_array(),
            radius: p.radius,
            mass: p.mass,
            fixed: p.fixed,
        }
    }
}

impl From<MpiParticle> for Particle {
    fn from(p: MpiParticle) -> Self {
        Self {
            id: p.id,
            position: Vec3::from_array(p.position),
            velocity: Vec3::from_array(p.velocity),
            acceleration: Vec3::from_array(p.acceleration),
            radius: p.radius,
            mass: p.mass,
            fixed: p.fixed,
            position_residual: Vec3::ZERO,
            velocity_residual: Vec3::ZERO,
        }
    }
}
