use glam::Vec3;
use rayon::prelude::*;
use crate::particle::Particle;
use crate::grid::Grid;
use crate::mesh::Mesh;

pub struct Simulation {
    pub particles: Vec<Particle>,
    pub meshes: Vec<Mesh>,
    pub grid: Grid,
    pub dt: f32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub gravity: Vec3,
    pub kn: f32, // Normal stiffness
    pub gn: f32, // Normal damping
}

impl Simulation {
    pub fn new(dt: f32, bounds_min: Vec3, bounds_max: Vec3, particle_count: usize) -> Self {
        let particles = Vec::with_capacity(particle_count);
        // Heuristic for cell size: 2 * max_radius. Let's assume max_radius = 0.1 for now.
        let cell_size = 0.2; 
        let grid = Grid::new(cell_size, bounds_min, bounds_max);

        Self {
            particles,
            meshes: Vec::new(),
            grid,
            dt,
            bounds_min,
            bounds_max,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            kn: 1e4, // Adjusted stiffness
            gn: 10.0,
        }
    }

    pub fn add_particle(&mut self, position: Vec3, radius: f32, mass: f32) {
        let id = self.particles.len();
        self.particles.push(Particle::new(id, position, radius, mass));
    }

    pub fn add_mesh(&mut self, mesh: Mesh) {
        self.meshes.push(mesh);
    }

    pub fn step(&mut self) {
        let dt = self.dt;

        // 0. Remove out of bounds particles
        let bounds_min = self.bounds_min;
        let bounds_max = self.bounds_max;
        self.particles.retain(|p| {
            p.position.y >= bounds_min.y - 1.0 && // Allow falling a bit below
            p.position.x >= bounds_min.x - 1.0 &&
            p.position.x <= bounds_max.x + 1.0 &&
            p.position.z >= bounds_min.z - 1.0 &&
            p.position.z <= bounds_max.z + 1.0
        });
        
        // Re-index
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.id = i;
        }

        // 1. First Half Update (Velocity Verlet)
        // v(t + 0.5dt) = v(t) + 0.5 * a(t) * dt
        // x(t + dt) = x(t) + v(t + 0.5dt) * dt
        for p in &mut self.particles {
            p.velocity += 0.5 * p.acceleration * dt;
            p.position += p.velocity * dt;
        }

        // 2. Update Grid
        self.grid.clear();
        for p in &self.particles {
            self.grid.insert(p);
        }

        // 3. Calculate Forces (at new positions)
        // We calculate the net force for each particle.
        // Since we can't easily mutate particles in parallel while reading them,
        // we compute the forces into a separate vector.
        let particles_ref = &self.particles;
        let grid_ref = &self.grid;
        let gravity = self.gravity;
        let bounds_min = self.bounds_min;
        let bounds_max = self.bounds_max;
        let kn = self.kn;
        let gn = self.gn;
        let meshes_ref = &self.meshes;

        let new_accelerations: Vec<Vec3> = particles_ref.par_iter().map(|p| {
            let mut force = gravity * p.mass;

            // Mesh Collisions
            for mesh in meshes_ref {
                for triangle in &mesh.triangles {
                    if let Some((penetration, normal)) = triangle.intersect_sphere(p.position, p.radius) {
                        let rel_vel = -p.velocity; // Mesh is stationary
                        let normal_vel = rel_vel.dot(normal);
                        // Repulsive force
                        let f_spring = kn * penetration * normal;
                        let f_dash = gn * normal_vel * normal;
                        force += f_spring + f_dash;
                    }
                }
            }

            // Wall collisions
            // Floor (Y-)
            if p.position.y - p.radius < bounds_min.y {
                let penetration = bounds_min.y - (p.position.y - p.radius);
                let normal = Vec3::Y;
                let rel_vel = -p.velocity; 
                let normal_vel = rel_vel.dot(normal);
                let f_spring = kn * penetration * normal; 
                let f_dash = gn * normal_vel * normal;
                force += f_spring + f_dash;
            }
            // Walls (X-, X+, Z-, Z+)
            // X-
            if p.position.x - p.radius < bounds_min.x {
                let penetration = bounds_min.x - (p.position.x - p.radius);
                let normal = Vec3::X;
                let rel_vel = -p.velocity;
                let normal_vel = rel_vel.dot(normal);
                force += (kn * penetration + gn * normal_vel) * normal;
            }
            // X+
            if p.position.x + p.radius > bounds_max.x {
                let penetration = (p.position.x + p.radius) - bounds_max.x;
                let normal = -Vec3::X;
                let rel_vel = -p.velocity;
                let normal_vel = rel_vel.dot(normal);
                force += (kn * penetration + gn * normal_vel) * normal;
            }
            // Z-
            if p.position.z - p.radius < bounds_min.z {
                let penetration = bounds_min.z - (p.position.z - p.radius);
                let normal = Vec3::Z;
                let rel_vel = -p.velocity;
                let normal_vel = rel_vel.dot(normal);
                force += (kn * penetration + gn * normal_vel) * normal;
            }
            // Z+
            if p.position.z + p.radius > bounds_max.z {
                let penetration = (p.position.z + p.radius) - bounds_max.z;
                let normal = -Vec3::Z;
                let rel_vel = -p.velocity;
                let normal_vel = rel_vel.dot(normal);
                force += (kn * penetration + gn * normal_vel) * normal;
            }

            // Particle-Particle collisions
            let neighbors = grid_ref.get_potential_collisions(p);
            for &other_idx in &neighbors {
                if p.id == other_idx { continue; }
                let other = &particles_ref[other_idx];
                
                let delta = p.position - other.position;
                let dist_sq = delta.length_squared();
                let rad_sum = p.radius + other.radius;
                
                if dist_sq < rad_sum * rad_sum && dist_sq > 0.00001 {
                    let dist = dist_sq.sqrt();
                    let normal = delta / dist;
                    let penetration = rad_sum - dist; 
                    
                    let rel_vel = p.velocity - other.velocity;
                    let normal_vel = rel_vel.dot(normal);
                    
                    // Force on p
                    // F = kn * penetration - gn * normal_vel
                    // Note: relative velocity is p - other. If moving towards each other, rel_vel dot normal is negative.
                    // Damping should oppose motion.
                    let f_spring = kn * penetration * normal;
                    let f_dash = -gn * normal_vel * normal;
                    
                    force += f_spring + f_dash;
                }
            }
            
            force / p.mass // Return acceleration
        }).collect();

        // 4. Second Half Update (Velocity Verlet)
        // v(t + dt) = v(t + 0.5dt) + 0.5 * a(t + dt) * dt
        for (p, a_new) in self.particles.iter_mut().zip(new_accelerations.into_iter()) {
            p.acceleration = a_new;
            p.velocity += 0.5 * p.acceleration * dt;
        }
    }
}
