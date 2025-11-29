struct Particle {
    position: vec3<f32>,
    _pad1: f32,
    velocity: vec3<f32>,
    _pad2: f32,
    acceleration: vec3<f32>,
    _pad3: f32,
    radius: f32,
    mass: f32,
    id: u32,
    _pad4: u32,
};

struct Params {
    bounds_min: vec4<f32>, // .w = dt
    bounds_max: vec4<f32>, // .w = padding
    periodic: vec4<u32>,   // .w = particle_count
    p_props: vec4<f32>,    // E, nu, rho, mu
    p_props2: vec4<f32>,   // e, pad, pad, pad
    w_props: vec4<f32>,    // E, nu, rho, mu
    w_props2: vec4<f32>,   // e, pad, pad, pad
    models: vec4<u32>,     // normal_id, tangential_id, pad, pad
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

fn effective_radius(r1: f32, r2: f32) -> f32 {
    return (r1 * r2) / (r1 + r2);
}

fn effective_mass(m1: f32, m2: f32) -> f32 {
    return (m1 * m2) / (m1 + m2);
}

fn effective_youngs(e1: f32, nu1: f32, e2: f32, nu2: f32) -> f32 {
    let inv_e = (1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2;
    return 1.0 / inv_e;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_count = params.periodic.w;
    let idx = global_id.x;
    if (idx >= particle_count) { return; }

    var p = particles[idx];
    let dt = params.bounds_min.w;

    // Force Calculation
    var force = vec3<f32>(0.0, -9.81 * p.mass, 0.0);

    // Particle-Particle (O(N^2))
    for (var i = 0u; i < particle_count; i++) {
        if (i == idx) { continue; }
        let other = particles[i];
        
        var dist_vec = p.position - other.position;
        let bounds_size = params.bounds_max.xyz - params.bounds_min.xyz;
        
        // MIC
        if (params.periodic.x != 0u) {
            if (dist_vec.x > bounds_size.x * 0.5) { dist_vec.x -= bounds_size.x; }
            else if (dist_vec.x < -bounds_size.x * 0.5) { dist_vec.x += bounds_size.x; }
        }
        if (params.periodic.y != 0u) {
            if (dist_vec.y > bounds_size.y * 0.5) { dist_vec.y -= bounds_size.y; }
            else if (dist_vec.y < -bounds_size.y * 0.5) { dist_vec.y += bounds_size.y; }
        }
        if (params.periodic.z != 0u) {
            if (dist_vec.z > bounds_size.z * 0.5) { dist_vec.z -= bounds_size.z; }
            else if (dist_vec.z < -bounds_size.z * 0.5) { dist_vec.z += bounds_size.z; }
        }
        
        let dist_sq = dot(dist_vec, dist_vec);
        let r_sum = p.radius + other.radius;
        
        if (dist_sq < r_sum * r_sum && dist_sq > 0.00001) {
            let dist = sqrt(dist_sq);
            let overlap = r_sum - dist;
            let normal = dist_vec / dist;
            let rel_vel = p.velocity - other.velocity;
            
            // Normal Force
            var total_fn = 0.0;
            let normal_model = params.models.x;
            
            if (normal_model == 1u) { // Hertzian
                // Hertzian
                let r_star = effective_radius(p.radius, other.radius);
                let m_star = effective_mass(p.mass, other.mass);
                let e_star = effective_youngs(params.p_props.x, params.p_props.y, params.p_props.x, params.p_props.y);
                
                let kn = 4.0 / 3.0 * e_star * sqrt(r_star);
                let force_elastic = kn * pow(overlap, 1.5);
                
                // Damping
                let restitution = params.p_props2.x;
                let ln_e = log(restitution);
                let beta = ln_e / sqrt(ln_e * ln_e + 3.14159 * 3.14159);
                let k_tangent = 2.0 * e_star * sqrt(r_star * overlap);
                let cn = 2.0 * sqrt(m_star * k_tangent) * abs(beta);
                
                let vn = dot(rel_vel, normal);
                let force_damping = -cn * vn;
                
                total_fn = max(force_elastic + force_damping, 0.0);
            } else { // Linear (Default/Fallback)
                // Linear Spring-Dashpot
                // Approx Kn, Gn
                let r_star = effective_radius(p.radius, other.radius);
                let m_star = effective_mass(p.mass, other.mass);
                let e_star = effective_youngs(params.p_props.x, params.p_props.y, params.p_props.x, params.p_props.y);
                
                let kn = e_star * r_star;
                let gn = 0.5 * sqrt(m_star * kn);
                
                let vn = dot(rel_vel, normal);
                total_fn = max(kn * overlap - gn * vn, 0.0);
            }
            
            force += total_fn * normal;
            
            // Tangential Force
            let tangential_model = params.models.y;
            let friction = params.p_props.w;
            let vt = rel_vel - dot(rel_vel, normal) * normal;
            
            if (length(vt) > 1e-6) {
                let ft_dir = normalize(vt);
                
                if (tangential_model == 0u) { // Coulomb
                     force -= friction * total_fn * ft_dir;
                } else { // Mindlin / Linear (Simplified to Coulomb for now in shader)
                     // Implementing full Mindlin history in shader is hard without extra buffers.
                     // Fallback to Coulomb for now.
                     force -= friction * total_fn * ft_dir;
                }
            }
        }
    }
    
    // Walls
    let bounds_min = params.bounds_min.xyz;
    let bounds_max = params.bounds_max.xyz;
    
    // Floor (Y-)
    if (params.periodic.y == 0u && p.position.y - p.radius < bounds_min.y) {
        let penetration = bounds_min.y - (p.position.y - p.radius);
        let normal = vec3<f32>(0.0, 1.0, 0.0);
        let rel_vel = -p.velocity;
        
        // Hertzian Wall
        let r_star = p.radius;
        let m_star = p.mass;
        let e_star = effective_youngs(params.p_props.x, params.p_props.y, params.w_props.x, params.w_props.y);
        
        let kn = 4.0 / 3.0 * e_star * sqrt(r_star);
        let force_elastic = kn * pow(penetration, 1.5);
        
        // Damping
        let restitution = params.p_props2.x;
        let ln_e = log(restitution);
        let beta = ln_e / sqrt(ln_e * ln_e + 3.14159 * 3.14159);
        let k_tangent = 2.0 * e_star * sqrt(r_star * penetration);
        let cn = 2.0 * sqrt(m_star * k_tangent) * abs(beta);
        
        let vn = dot(rel_vel, normal);
        let force_damping = -cn * vn;
        
        let total_fn = max(force_elastic + force_damping, 0.0);
        force += total_fn * normal;
        
        // Friction
        let friction = params.p_props.w;
        let vt = rel_vel - vn * normal;
        if (length(vt) > 1e-6) {
             let ft_dir = normalize(vt);
             force -= friction * total_fn * ft_dir;
        }
    }
    
    // Other walls...
    
    p.acceleration = force / p.mass;
    
    // Integration (Symplectic Euler)
    p.velocity += p.acceleration * dt;
    p.position += p.velocity * dt;
    
    // Wrap
    let bounds_size = params.bounds_max.xyz - params.bounds_min.xyz;
    if (params.periodic.x != 0u) {
        if (p.position.x < bounds_min.x) { p.position.x += bounds_size.x; }
        else if (p.position.x >= bounds_max.x) { p.position.x -= bounds_size.x; }
    }
    if (params.periodic.y != 0u) {
        if (p.position.y < bounds_min.y) { p.position.y += bounds_size.y; }
        else if (p.position.y >= bounds_max.y) { p.position.y -= bounds_size.y; }
    }
    if (params.periodic.z != 0u) {
        if (p.position.z < bounds_min.z) { p.position.z += bounds_size.z; }
        else if (p.position.z >= bounds_max.z) { p.position.z -= bounds_size.z; }
    }
    
    particles[idx] = p;
}
