use glam::Vec3;
use crate::contact::Contact;

/// Available normal force models.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormalForceModel {
    /// Linear Spring-Dashpot model.
    LinearSpringDashpot,
    /// Hertzian elastic contact model.
    Hertzian,
    /// Hysteretic linear spring model (simple plasticity).
    Hysteretic,
    /// Johnson-Kendall-Roberts (JKR) adhesion model.
    JKR,
    /// Simplified JKR (sJKR) explicit approximation.
    SimplifiedJKR,
}

/// Available tangential force models.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TangentialForceModel {
    /// Coulomb friction model.
    Coulomb,
    /// Linear Spring-Coulomb model (simplified).
    LinearSpringCoulomb,
    /// Mindlin-Deresiewicz no-slip model.
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
/// Computes the normal force using a Linear Spring-Dashpot model.
///
/// # Arguments
///
/// * `overlap` - The overlap distance between particles (positive for penetration).
/// * `normal` - The contact normal vector.
/// * `relative_velocity` - The relative velocity vector.
/// * `kn` - Normal stiffness.
/// * `gn` - Normal damping coefficient.
pub fn linear_spring_dashpot(
    overlap: f32,
    normal: Vec3,
    relative_velocity: Vec3,
    kn: f32,
    gn: f32,
) -> Vec3 {
    let vn = relative_velocity.dot(normal);
    // Force is repulsive, so positive magnitude pushes particles apart?
    // Usually: F = -k*delta - c*v
    // If overlap is positive (penetration), force should be repulsive (positive along normal).
    // Wait, if normal points from B to A, and we want force on A.
    // Let's assume normal points from other to self.
    // If approaching (vn < 0), damping adds to stiffness? No, damping resists motion.
    // If approaching, force should be higher to stop it.
    // So if vn < 0, we want more repulsive force.
    // So -gn * vn (which is positive) adds to force.
    
    let fn_mag = kn * overlap - gn * vn;
    let fn_mag = fn_mag.max(0.0); // No attraction
    fn_mag * normal
}

// Hertzian Contact
/// Computes the normal force using the Hertzian contact model.
///
/// Includes non-linear elasticity and damping.
///
/// # Arguments
///
/// * `overlap` - The overlap distance.
/// * `normal` - The contact normal.
/// * `relative_velocity` - The relative velocity.
/// * `e_star` - Effective Young's modulus.
/// * `r_star` - Effective radius.
/// * `m_star` - Effective mass.
/// * `restitution_coefficient` - Coefficient of restitution.
pub fn hertzian_contact(
    overlap: f32,
    normal: Vec3,
    relative_velocity: Vec3,
    e_star: f32,
    r_star: f32,
    m_star: f32,
    restitution_coefficient: f32,
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
    
    let ln_e = restitution_coefficient.ln();
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

// JKR Contact Model
pub fn compute_jkr_force(
    overlap: f32,
    normal: Vec3,
    _relative_velocity: Vec3, // Damping not specified in JKR, usually added separately
    e_star: f32,
    r_star: f32,
    surface_energy: f32,
) -> Vec3 {
    // JKR equations provided by user:
    // delta = a^2 / R* - sqrt(8 * pi * gamma * a / E*)
    // F = 4 * E* * a^3 / (3 * R*) - sqrt(8 * pi * gamma * E* * a^3)
    // Pull-off force: F_pull = -2/3 * pi * gamma * R*
    
    // We need to solve for a (contact radius) given overlap (delta).
    // Equation: f(a) = a^2 / R* - sqrt(8 * pi * gamma * a / E*) - delta = 0
    // Let C = 8 * pi * gamma / E*
    // f(a) = a^2 / R* - sqrt(C * a) - delta = 0
    // f'(a) = 2*a / R* - 0.5 * sqrt(C) * a^(-1/2)
    
    // Newton-Raphson iteration
    // Initial guess: Hertzian a0 = sqrt(R* * delta) (if delta > 0)
    // If delta <= 0, we need a better guess. JKR allows contact for negative overlap.
    // For delta < 0, a is small but positive.
    
    let gamma = surface_energy;
    if gamma <= 0.0 {
        // Fallback to Hertzian if no adhesion
        // We don't have mass here for damping, so just elastic part
        let kn = 4.0 / 3.0 * e_star * r_star.sqrt();
        let f_mag = if overlap > 0.0 { kn * overlap.powf(1.5) } else { 0.0 };
        return f_mag * normal;
    }

    let c_val = 8.0 * std::f32::consts::PI * gamma / e_star;
    let sqrt_c = c_val.sqrt();
    
    let mut a = if overlap > 0.0 {
        (r_star * overlap).sqrt()
    } else {
        // Safe guess for negative overlap
        r_star * 0.5
    };
    
    // Iteration
    for _ in 0..20 {
        if a <= 0.0 { a = 1e-9; } 
        
        let term_sqrt = (c_val * a).sqrt();
        let f_val = a * a / r_star - term_sqrt - overlap;
        let df_val = 2.0 * a / r_star - 0.5 * sqrt_c / a.sqrt();
        
        if df_val.abs() < 1e-9 { break; }
        
        let delta_a = f_val / df_val;
        let max_step = a * 0.5;
        let step = delta_a.clamp(-max_step, max_step);
        a -= step;
        
        if step.abs() < 1e-6 * r_star { break; }
    }
    
    // Check convergence
    let term_sqrt = (c_val * a).sqrt();
    let f_val = a * a / r_star - term_sqrt - overlap;
    if f_val.abs() > 1e-3 * r_star {
        // No root found (likely separated beyond pull-off)
        return Vec3::ZERO;
    }

    if a <= 0.0 {
        return Vec3::ZERO;
    }
    
    // Calculate Force
    let term1 = 4.0 * e_star * a.powi(3) / (3.0 * r_star);
    let term2 = (8.0 * std::f32::consts::PI * gamma * e_star * a.powi(3)).sqrt();
    
    let force_mag = term1 - term2;
    
    // Check pull-off limit
    let f_pull = -2.0 / 3.0 * std::f32::consts::PI * gamma * r_star;
    
    if force_mag < f_pull {
        return Vec3::ZERO;
    }
    
    force_mag * normal
}

/// Computes the normal force using the Simplified JKR (sJKR) model.
///
/// An explicit, non-iterative approximation of JKR.
/// Uses a "cohesive Hertz" approach with hysteresis based on relative velocity.
///
/// # Arguments
///
/// * `overlap` - The overlap distance.
/// * `normal` - The contact normal.
/// * `relative_velocity` - The relative velocity.
/// * `e_star` - Effective Young's modulus.
/// * `r_star` - Effective radius.
/// * `surface_energy` - Surface energy (gamma).
pub fn compute_sjkr_force(
    overlap: f32,
    normal: Vec3,
    relative_velocity: Vec3,
    e_star: f32,
    r_star: f32,
    surface_energy: f32,
) -> Vec3 {
    let gamma = surface_energy;
    if gamma <= 0.0 {
        // Fallback to Hertzian
        let kn = 4.0 / 3.0 * e_star * r_star.sqrt();
        let f_mag = if overlap > 0.0 { kn * overlap.powf(1.5) } else { 0.0 };
        return f_mag * normal;
    }

    // Critical pull-off parameters
    // F_pull_magnitude = 1.5 * pi * gamma * R*
    let f_pull = 1.5 * std::f32::consts::PI * gamma * r_star;
    
    // delta_c = -sqrt(3 * pi^2 * gamma^2 * R* / E*^2)
    // term inside sqrt: 3 * pi^2 * gamma^2 * R* / E*^2
    let term = 3.0 * std::f32::consts::PI.powi(2) * gamma.powi(2) * r_star / e_star.powi(2);
    let delta_c = -term.sqrt();
    
    // Check detachment
    if overlap < delta_c {
        return Vec3::ZERO;
    }
    
    // Determine state (Loading vs Unloading)
    // We use relative velocity along normal.
    // v_n = v_rel . n
    // If v_n < 0, particles are approaching (Loading).
    // If v_n > 0, particles are separating (Unloading).
    // Note: relative_velocity is v_a - v_b. 
    // If particles move towards each other, v_a moves to right, v_b to left?
    // Let's check convention in simulation.rs:
    // let rel_vel = p.velocity - other.velocity ...
    // let normal = dist_vec / dist; (points from other to p)
    // vn = rel_vel.dot(normal)
    // If p moves towards other (opposing normal), vn < 0.
    // So vn < 0 is approaching (Loading).
    // vn > 0 is separating (Unloading).
    
    let vn = relative_velocity.dot(normal);
    let is_unloading = vn > 0.0;
    
    // Coherency parameter (0 for loading, 1 for unloading)
    // We can also use a smooth transition if needed, but sJKR usually implies a switch.
    let coherency = if is_unloading { 1.0 } else { 0.0 };
    
    // Hertzian force part
    // F_Hertz = 4/3 * E* * sqrt(R*) * delta^1.5
    // Note: delta can be negative in sJKR (necking).
    // Standard Hertz is 0 for delta < 0.
    // However, for sJKR unloading, we extend the curve?
    // "F_n = F_Hertz - F_pull_magnitude"
    // If delta < 0, F_Hertz is undefined (complex) or 0?
    // Usually sJKR assumes contact area exists until delta_c.
    // But Hertzian formula a = sqrt(R*delta) implies delta > 0.
    // The "Cohesive Hertz" model often uses:
    // F = K * delta^1.5 - F_pull (for unloading)
    // But if delta < 0, this doesn't work directly.
    // Maybe the approximation is:
    // F = F_Hertz(delta) - coherency * F_pull
    // And for delta < 0, F_Hertz is 0, so F = -F_pull.
    // This gives a constant adhesive force for negative overlap?
    // Or does it follow a different curve?
    // The user said: "F_n = F_Hertz - F_pull_magnitude ... clamped to 0 if delta is negative"
    // So F_Hertz is 0 if delta < 0.
    // Thus for delta < 0 (but > delta_c), F = -F_pull.
    // This creates a constant pull-off force region.
    
    let f_hertz = if overlap > 0.0 {
        4.0 / 3.0 * e_star * r_star.sqrt() * overlap.powf(1.5)
    } else {
        0.0
    };
    
    let f_adhesion = coherency * f_pull;
    
    let total_force = f_hertz - f_adhesion;
    
    // Ensure we don't return negative force if we are loading?
    // Loading: F = F_Hertz. (No adhesion).
    // Unloading: F = F_Hertz - F_pull.
    
    // What if total_force is negative during unloading? That's expected (attraction).
    
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
