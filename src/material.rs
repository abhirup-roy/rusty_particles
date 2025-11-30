#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub youngs_modulus: f32, // E
    pub poissons_ratio: f32, // nu
    pub density: f32,        // rho
    pub friction_coefficient: f32, // mu
    pub restitution_coefficient: f32, // e
    pub surface_energy: f32, // gamma
}

impl Material {
    pub fn new(youngs_modulus: f32, poissons_ratio: f32, density: f32, friction_coefficient: f32, restitution_coefficient: f32, surface_energy: f32) -> Self {
        Self {
            youngs_modulus,
            poissons_ratio,
            density,
            friction_coefficient,
            restitution_coefficient,
            surface_energy,
        }
    }
}

// Helper to compute effective properties between two materials
pub fn effective_radius(r1: f32, r2: f32) -> f32 {
    (r1 * r2) / (r1 + r2)
}

pub fn effective_mass(m1: f32, m2: f32) -> f32 {
    (m1 * m2) / (m1 + m2)
}

pub fn effective_youngs_modulus(e1: f32, nu1: f32, e2: f32, nu2: f32) -> f32 {
    // 1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2
    let inv_e_star = (1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2;
    1.0 / inv_e_star
}

pub fn effective_shear_modulus(_g1: f32, _g2: f32) -> f32 {
    // 1/G* = (2-nu1)/G1 + (2-nu2)/G2 ? No, usually:
    // 1/G* = 1/G1 + 1/G2 for some models, or derived from E and nu.
    // G = E / (2(1+nu))
    // For Mindlin: 1/G* = (2-nu1)/G1 + (2-nu2)/G2 is sometimes used, but standard is:
    // 1/G* = (1/G1 + 1/G2) ?
    // Let's stick to calculating G for each and then combining.
    // Hertz/Mindlin usually uses E* and G*.
    // 1/G* = (2 - nu1)/(4G1) + (2 - nu2)/(4G2) ?
    // Let's implement G calculation first.
    0.0 // Placeholder
}

pub fn shear_modulus(e: f32, nu: f32) -> f32 {
    e / (2.0 * (1.0 + nu))
}
