use glam::Vec3;
use crate::material::{Material, effective_radius, effective_mass, effective_youngs_modulus, shear_modulus};
use crate::contact::Contact;

#[derive(Clone, Copy, Debug)]
pub enum NormalForceModel {
    LinearSpringDashpot,
    Hertzian,
    Hysteretic,
}

#[derive(Clone, Copy, Debug)]
pub enum TangentialForceModel {
    Coulomb,
    LinearSpringCoulomb,
    Mindlin,
}

pub struct InteractionParams {
    pub kn: f32, // Normal stiffness
    pub kt: f32, // Tangential stiffness
    pub gn: f32, // Normal damping
    pub gt: f32, // Tangential damping
    pub mu: f32, // Friction coefficient
    pub e: f32,  // Restitution coefficient
}

// Linear Spring-Dashpot (Cundall & Strack)
pub fn linear_spring_dashpot(
    overlap: f32,
    normal: Vec3,
    relative_velocity: Vec3,
    kn: f32,
    gn: f32,
) -> Vec3 {
    let vn = relative_velocity.dot(normal);
    let force_mag = -kn * overlap - gn * vn;
    // Force is repulsive, so positive magnitude pushes particles apart?
    // Usually: F = -k*delta - c*v
    // If overlap is positive (penetration), force should be repulsive (positive along normal).
    // Wait, if normal points from B to A, and we want force on A.
    // Let's assume normal points from other to self.
    // Force = (kn * overlap - gn * vn) * normal
    // If overlap > 0, we want repulsion.
    // Let's stick to: F_n = (kn * delta + gn * vn_mag) * n
    // But usually damping opposes velocity.
    
    let fn_mag = kn * overlap + gn * vn; // vn is negative if approaching
    // If approaching (vn < 0), damping adds to stiffness? No, damping resists motion.
    // If approaching, force should be higher to stop it.
    // So if vn < 0, we want more repulsive force.
    // So -gn * vn (which is positive) adds to force.
    
    let fn_mag = kn * overlap - gn * vn;
    let fn_mag = fn_mag.max(0.0); // No attraction
    fn_mag * normal
}

// Hertzian Contact
pub fn hertzian_contact(
    overlap: f32,
    normal: Vec3,
    relative_velocity: Vec3,
    e_star: f32,
    r_star: f32,
    m_star: f32,
    restitution_coeff: f32,
) -> Vec3 {
    if overlap <= 0.0 {
        return Vec3::ZERO;
    }
    
    // Kn depends on overlap
    // F = 4/3 * E* * sqrt(R*) * delta^(3/2)
    let kn = 4.0 / 3.0 * e_star * r_star.sqrt();
    let force_elastic = kn * overlap.powf(1.5);
    
    // Damping for Hertz:
    // Beta = ln(e) / sqrt(ln(e)^2 + pi^2)
    // Cn = 2 * sqrt(m* * kn(delta)) * beta
    // But kn depends on delta.
    // Usually simplified or derived.
    // Tsuji et al (1992): F = K * delta^1.5 + C * delta^0.25 * v_n
    
    let ln_e = restitution_coeff.ln();
    let beta = ln_e / (ln_e.powi(2) + std::f32::consts::PI.powi(2)).sqrt();
    
    // Stiffness at this overlap
    // K_hertz = 4/3 E* sqrt(R*)
    // Stiffness tangent = dF/ddelta = 2 * E* * sqrt(R* * delta)
    // Damping coeff depends on stiffness.
    // cn = 2 * m* * sqrt(K_tangent / m*) * beta ?
    // cn = 2 * sqrt(m* * K_tangent) * beta
    
    let k_tangent = 2.0 * e_star * (r_star * overlap).sqrt();
    let cn = 2.0 * (m_star * k_tangent).sqrt() * beta.abs(); // beta is negative
    
    let vn = relative_velocity.dot(normal);
    let force_damping = -cn * vn;
    
    let total_force = (force_elastic + force_damping).max(0.0);
    total_force * normal
}

// Hysteretic Linear Spring (Walton & Braun)
// Requires state (max overlap or similar).
// Loading stiffness K1, Unloading stiffness K2.
// K2 > K1 implies energy loss.
// F = K1 * delta (loading)
// F = K2 * (delta - delta_0) (unloading)
// We need to store state.
pub fn hysteretic_contact(
    overlap: f32,
    normal: Vec3,
    _contact: &mut Contact,
    k1: f32,
    _k2: f32,
) -> Vec3 {
    // We need to know if we are loading or unloading.
    // Or we track the force history.
    // Walton & Braun:
    // If delta is increasing, K = K1.
    // If delta is decreasing, K = K2.
    // But we need to know the turning point.
    // Simplified: Store the maximum overlap achieved?
    // Actually, usually we store the current force or similar.
    // Let's assume we store `normal_force_history`.
    
    // If new overlap > old overlap (approx), loading?
    // Better: Update force based on displacement increment.
    // But we don't have displacement increment easily here without storing prev overlap.
    // Let's store `prev_overlap` in Contact?
    // For now, let's implement a simpler version or assume we add `prev_overlap` to Contact.
    
    // Let's assume K1 for loading, K2 for unloading.
    // Effective restitution e = sqrt(K1/K2).
    
    // If current force is F_old.
    // F_new = F_old + K * (delta_new - delta_old).
    // This requires delta_old.
    
    // Let's stick to Linear Spring-Dashpot as default and Hertzian.
    // Hysteretic might require more state changes.
    // I'll leave a placeholder or implement if I add `prev_overlap`.
    
    // Placeholder for now, fall back to linear.
    let _vn = 0.0; // Need velocity for direction?
    // If vn < 0 (approaching), use K1.
    // If vn > 0 (separating), use K2.
    // But we need to ensure continuity.
    // F = K1 * delta if loading.
    // F = F_max - K2 * (delta_max - delta) if unloading.
    // This requires storing delta_max.
    
    // Let's add `max_overlap` to Contact struct later if needed.
    // For now, return simple linear.
    linear_spring_dashpot(overlap, normal, Vec3::ZERO, k1, 0.0)
}

// Tangential Forces

// Coulomb Limit
pub fn coulomb_friction(
    normal_force_mag: f32,
    relative_velocity: Vec3,
    normal: Vec3,
    mu: f32,
) -> Vec3 {
    let vt = relative_velocity - relative_velocity.dot(normal) * normal;
    if vt.length_squared() < 1e-8 {
        return Vec3::ZERO;
    }
    let vt_dir = vt.normalize();
    -mu * normal_force_mag * vt_dir
}

// Linear Spring with Coulomb Limit
pub fn linear_spring_coulomb(
    normal_force_mag: f32,
    relative_velocity: Vec3,
    normal: Vec3,
    contact: &mut Contact,
    kt: f32,
    mu: f32,
    dt: f32,
) -> Vec3 {
    // Relative tangential velocity
    let vt = relative_velocity - relative_velocity.dot(normal) * normal;
    
    // Integrate tangential displacement
    // delta_t += v_t * dt
    // But we must rotate the stored displacement if the contact plane rotates.
    // For simplicity, assume small rotation or ignore for now (common in simple DEM).
    
    contact.tangential_displacement += vt * dt;
    
    // Trial force
    let ft_trial = -kt * contact.tangential_displacement;
    let ft_mag = ft_trial.length();
    let f_coulomb = mu * normal_force_mag;
    
    if ft_mag > f_coulomb {
        // Slip
        let ft_dir = if ft_mag > 1e-6 { ft_trial / ft_mag } else { Vec3::ZERO };
        let ft = f_coulomb * ft_dir;
        // Adjust displacement to match force
        contact.tangential_displacement = -ft / kt;
        ft
    } else {
        ft_trial
    }
}

// Mindlin-Deresiewicz
// F_t depends on loading history and normal force.
// dF_t / d_delta_t = 8 * G* * a * theta
// complicated.
// Usually simplified:
// Kt = 8 * G* * sqrt(R* * delta_n)
// Then use spring-coulomb with this variable Kt.
pub fn mindlin_contact(
    normal_force_mag: f32,
    overlap: f32,
    relative_velocity: Vec3,
    normal: Vec3,
    contact: &mut Contact,
    g_star: f32,
    r_star: f32,
    mu: f32,
    dt: f32,
) -> Vec3 {
    // Kt depends on overlap
    // Kt = 8 * G* * sqrt(R* * delta_n)
    let kt = 8.0 * g_star * (r_star * overlap).sqrt();
    
    linear_spring_coulomb(normal_force_mag, relative_velocity, normal, contact, kt, mu, dt)
}
