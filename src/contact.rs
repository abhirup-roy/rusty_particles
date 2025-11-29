use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub tangential_displacement: Vec3, // Accumulated slip
    pub normal_force_history: f32,     // For hysteretic models
    pub age: usize,                    // Number of steps active
}

impl Contact {
    pub fn new() -> Self {
        Self {
            tangential_displacement: Vec3::ZERO,
            normal_force_history: 0.0,
            age: 0,
        }
    }
}
