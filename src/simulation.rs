use glam::Vec3;
use rayon::prelude::*;
use crate::particle::Particle;
use crate::grid::Grid;
use crate::mesh::Mesh;
use crate::material::Material;
use crate::contact::Contact;
use crate::gpu;
use dashmap::DashMap;

pub struct Simulation {
    pub particles: Vec<Particle>,
    pub grid: Grid,
    pub dt: f32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub meshes: Vec<Mesh>,
    pub contacts: DashMap<(usize, usize), Contact>,
    pub material: Material, // Global material for now, or per particle?
    // User asked for "specify the young's modulus and density of walls and particles seperately"
    // So we need materials for particles and walls.
    // Let's assume all particles share a material, and walls share a material for now.
    pub particle_material: Material,
    pub wall_material: Material,
    pub periodic: [bool; 3],
    pub gpu_sim: Option<gpu::GpuSimulation>,
}

impl Simulation {
    pub fn new(dt: f32, bounds_min: Vec3, bounds_max: Vec3, particle_count: usize) -> Self {
        let mut particles = Vec::with_capacity(particle_count);
        // Heuristic for cell size: 2 * max_radius. Let's assume max_radius = 0.1 for now.
        // Heuristic for cell size: 2 * max_radius. Let's assume max_radius = 0.1 for now.
        let grid = Grid::new(1.0, bounds_min, bounds_max); // Cell size will be updated
        
        // Default materials
        let particle_material = Material::new(1e7, 0.3, 2500.0, 0.5, 0.5);
        let wall_material = Material::new(1e9, 0.3, 7800.0, 0.5, 0.5);

        Self {
            particles,
            grid,
            dt,
            bounds_min,
            bounds_max,
            meshes: Vec::new(),
            contacts: DashMap::new(),
            material: particle_material, // Deprecated
            particle_material,
            wall_material,
            periodic: [false; 3],
            gpu_sim: None,
        }
    }

    pub fn add_particle(&mut self, position: Vec3, radius: f32, mass: f32) {
        let id = self.particles.len();
        self.particles.push(Particle::new(id, position, radius, mass));
    }

    pub fn add_mesh(&mut self, mesh: Mesh) {
        self.meshes.push(mesh);
    }

    pub fn enable_gpu(&mut self) {
        // Initialize GPU simulation
        // This requires async, but we are in sync context.
        // We use pollster to block.
        let gpu_sim = pollster::block_on(gpu::GpuSimulation::new(
            &self.particles,
            self.dt,
            self.bounds_min,
            self.bounds_max,
            self.periodic,
            self.particle_material,
            self.wall_material,
        ));
        self.gpu_sim = Some(gpu_sim);
    }

    pub fn step(&mut self) {
        if let Some(gpu_sim) = &mut self.gpu_sim {
            gpu_sim.step();
            // Sync back particles for VTK or other logic?
            // Doing it every step is slow.
            // But Simulation struct owns particles.
            // If we want to write VTK, we need to sync.
            // For now, let's NOT sync every step, only when needed (e.g. before write_vtk).
            // But step() updates self.particles in CPU mode.
            // If we switch modes, we need to sync.
            // Let's assume if GPU is enabled, we run on GPU.
            // We should add a sync_from_gpu() method.
        } else {
            self.step_cpu();
        }
    }

    fn step_cpu(&mut self) {
        let dt = self.dt;

        // 0. Handle boundaries (Wrap or Remove)
        let bounds_min = self.bounds_min;
        let bounds_max = self.bounds_max;
        let bounds_size = bounds_max - bounds_min;
        let periodic = self.periodic;
        
        if periodic[0] || periodic[1] || periodic[2] {
            // Wrap particles
            self.particles.par_iter_mut().for_each(|p| {
                if periodic[0] {
                    if p.position.x < bounds_min.x { p.position.x += bounds_size.x; }
                    else if p.position.x >= bounds_max.x { p.position.x -= bounds_size.x; }
                }
                if periodic[1] {
                    if p.position.y < bounds_min.y { p.position.y += bounds_size.y; }
                    else if p.position.y >= bounds_max.y { p.position.y -= bounds_size.y; }
                }
                if periodic[2] {
                    if p.position.z < bounds_min.z { p.position.z += bounds_size.z; }
                    else if p.position.z >= bounds_max.z { p.position.z -= bounds_size.z; }
                }
            });
            
            // Remove out of bounds for non-periodic dimensions
             self.particles.retain(|p| {
                (periodic[0] || (p.position.x >= bounds_min.x - 1.0 && p.position.x <= bounds_max.x + 1.0)) &&
                (periodic[1] || (p.position.y >= bounds_min.y - 1.0)) && // Allow falling a bit below if not periodic y
                (periodic[2] || (p.position.z >= bounds_min.z - 1.0 && p.position.z <= bounds_max.z + 1.0))
            });
        } else {
             // Original removal logic
            self.particles.retain(|p| {
                p.position.y >= bounds_min.y - 1.0 &&
                p.position.x >= bounds_min.x - 1.0 &&
                p.position.x <= bounds_max.x + 1.0 &&
                p.position.z >= bounds_min.z - 1.0 &&
                p.position.z <= bounds_max.z + 1.0
            });
        }
        
        // Re-index
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.id = i;
        }

        // 1. First Half Update (Velocity Verlet)
        // v(t + 0.5dt) = v(t) + 0.5 * a(t) * dt
        // x(t + dt) = x(t) + v(t + 0.5dt) * dt
        self.particles.par_iter_mut().for_each(|p| {
            p.velocity += 0.5 * p.acceleration * dt;
            p.position += p.velocity * dt;
        });

        // 2. Force Calculation
        // Reset forces (gravity)
        let g = Vec3::new(0.0, -9.81, 0.0);
        self.particles.par_iter_mut().for_each(|p| {
            p.acceleration = g;
        });

        // Broad-phase
        self.grid.clear();
        for p in &self.particles {
            self.grid.insert(p);
        }

        // Narrow-phase & Solve
        let p_mat = self.particle_material;
        let w_mat = self.wall_material;
        
        use rayon::prelude::*;
        use crate::physics::*;
        use crate::material::*;
        
        let grid = &self.grid;
        let particles = &self.particles;
        let contacts = &self.contacts;
        let meshes = &self.meshes;
        
        let forces: Vec<Vec3> = self.particles.par_iter().enumerate().map(|(i, p)| {
            let mut f_total = Vec3::ZERO;
            
            // Neighbor search
            let neighbors = grid.get_potential_collisions(p, periodic);
            for &j in &neighbors {
                if i == j { continue; }
                let other = &particles[j];
                
                let mut dist_vec = p.position - other.position;
                
                // Minimum Image Convention
                if periodic[0] {
                    if dist_vec.x > bounds_size.x * 0.5 { dist_vec.x -= bounds_size.x; }
                    else if dist_vec.x < -bounds_size.x * 0.5 { dist_vec.x += bounds_size.x; }
                }
                if periodic[1] {
                    if dist_vec.y > bounds_size.y * 0.5 { dist_vec.y -= bounds_size.y; }
                    else if dist_vec.y < -bounds_size.y * 0.5 { dist_vec.y += bounds_size.y; }
                }
                if periodic[2] {
                    if dist_vec.z > bounds_size.z * 0.5 { dist_vec.z -= bounds_size.z; }
                    else if dist_vec.z < -bounds_size.z * 0.5 { dist_vec.z += bounds_size.z; }
                }
                
                let dist_sq = dist_vec.length_squared();
                let r_sum = p.radius + other.radius;
                
                if dist_sq < r_sum * r_sum {
                    let dist = dist_sq.sqrt();
                    let overlap = r_sum - dist;
                    let normal = if dist > 1e-6 { dist_vec / dist } else { Vec3::Y };
                    let rel_vel = p.velocity - other.velocity;
                    
                    // Material properties
                    let r_star = effective_radius(p.radius, other.radius);
                    let m_star = effective_mass(p.mass, other.mass);
                    let e_star = effective_youngs_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio, p_mat.youngs_modulus, p_mat.poissons_ratio);
                    let g_star = effective_shear_modulus(shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio), shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio));
                    
                    // Contact state
                    // Key must be sorted to be unique for pair
                    let key = if i < j { (i, j) } else { (j, i) };
                    // We need to access DashMap.
                    // Since we are in a read-only pass over particles (conceptually), we can write to DashMap.
                    // But we are inside a par_iter map.
                    
                    // Note: DashMap might deadlock if we try to upgrade/downgrade?
                    // entry() is safe.
                    
                    let mut contact = contacts.entry(key).or_insert(Contact::new());
                    contact.age += 1;
                    
                    // Normal Force (Hertzian)
                    let f_normal = hertzian_contact(overlap, normal, rel_vel, e_star, r_star, m_star, p_mat.restitution_coefficient);
                    let fn_mag = f_normal.length();
                    
                    // Tangential Force (Mindlin)
                    let f_tangent = mindlin_contact(fn_mag, overlap, rel_vel, normal, &mut contact, g_star, r_star, p_mat.friction_coefficient, dt);
                    
                    f_total += f_normal + f_tangent;
                }
            }
            
            // Particle-Mesh (Walls)
            // If periodic, we might skip wall collisions on periodic boundaries?
            // Usually periodic implies no walls on those sides.
            // But we might have internal meshes.
            // Let's assume meshes are internal or user handles placement.
            // However, explicit wall checks (like floor) need to be conditional.
            
            for mesh in meshes {
                for triangle in &mesh.triangles {
                    if let Some((penetration, normal)) = triangle.intersect_sphere(p.position, p.radius) {
                        let rel_vel = p.velocity; // Wall is static
                        
                        // Wall material
                        let r_star = p.radius; // Wall radius is infinite
                        let m_star = p.mass; // Wall mass infinite
                        let e_star = effective_youngs_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio, w_mat.youngs_modulus, w_mat.poissons_ratio);
                        let g_star = effective_shear_modulus(shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio), shear_modulus(w_mat.youngs_modulus, w_mat.poissons_ratio));
                        
                        // Hertzian normal force
                        let f_normal = hertzian_contact(penetration, normal, rel_vel, e_star, r_star, m_star, p_mat.restitution_coefficient);
                        
                        // Simple Coulomb for wall
                        let fn_mag = f_normal.length();
                        let f_tangent = coulomb_friction(fn_mag, rel_vel, normal, p_mat.friction_coefficient);
                        
                        f_total += f_normal + f_tangent;
                    }
                }
            }
            
            f_total
        }).collect();
        
        // Apply forces
        self.particles.par_iter_mut().zip(forces.par_iter()).for_each(|(p, f)| {
            p.acceleration += *f / p.mass;
        });

        // 3. Second Half Update (Velocity Verlet)
        // v(t + dt) = v(t + 0.5dt) + 0.5 * a(t + dt) * dt
        self.particles.par_iter_mut().for_each(|p| {
            p.velocity += 0.5 * p.acceleration * dt;
        });
    }

    pub fn sync_particles(&mut self) {
        if let Some(gpu_sim) = &self.gpu_sim {
            self.particles = gpu_sim.read_particles();
        }
    }
}
