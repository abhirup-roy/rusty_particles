pub mod particle;
pub mod grid;
pub mod simulation;
pub mod vtk;
pub mod mesh;
pub mod material;
pub mod contact;
pub mod physics;
pub mod gpu;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use glam::Vec3;

#[pyfunction]
fn set_num_threads(num_threads: usize) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to set num threads: {}", e)))?;
    Ok(())
}

#[pymodule]
fn rusty_particles(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySimulation>()?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    
    #[pyfn(m)]
    fn hello() -> PyResult<String> {
        Ok("Hello from Rust!".to_string())
    }
    
    Ok(())
}

/// Python wrapper for the Rust simulation engine.
///
/// This class exposes the core simulation functionality to Python, allowing
/// users to create, configure, and run particle simulations.
#[pyclass(name = "Simulation")]
struct PySimulation {
    inner: Box<simulation::Simulation>,
}

#[pymethods]
impl PySimulation {
    /// Creates a new simulation instance.
    ///
    /// # Arguments
    ///
    /// * `dt` - The time step for the simulation (seconds).
    /// * `min_x`, `min_y`, `min_z` - The minimum coordinates of the simulation bounds.
    /// * `max_x`, `max_y`, `max_z` - The maximum coordinates of the simulation bounds.
    /// * `particle_count` - The initial capacity for particles.
    #[staticmethod]
    fn create(dt: f32, min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32, particle_count: usize) -> Self {
        let min = Vec3::new(min_x, min_y, min_z);
        let max = Vec3::new(max_x, max_y, max_z);
        
        let sim = simulation::Simulation::new(dt, min, max, particle_count);
        
        Self {
            inner: Box::new(sim),
        }
    }

    /// Adds a particle to the simulation.
    ///
    /// # Arguments
    ///
    /// * `x`, `y`, `z` - The position of the particle.
    /// * `radius` - The radius of the particle.
    /// * `mass` - The mass of the particle.
    /// * `fixed` - Whether the particle is fixed in space (immovable).
    #[pyo3(signature = (x, y, z, radius, mass, fixed=false))]
    fn add_particle(&mut self, x: f32, y: f32, z: f32, radius: f32, mass: f32, fixed: bool) {
        let pos = Vec3::new(x, y, z);
        self.inner.add_particle(pos, radius, mass, fixed);
    }

    /// Adds a mesh from an STL file to the simulation.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the STL file.
    fn add_mesh(&mut self, path: String) -> PyResult<()> {
        let mesh = mesh::Mesh::load_stl(&path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        self.inner.add_mesh(mesh);
        Ok(())
    }

    /// Advances the simulation by one time step.
    fn step(&mut self) {
        self.inner.step();
    }

    #[pyo3(signature = (youngs_modulus, poissons_ratio, density, friction_coefficient, restitution_coefficient, surface_energy=0.0))]
    /// Sets the material properties for all particles.
    ///
    /// # Arguments
    ///
    /// * `youngs_modulus` - Young's modulus (Pa).
    /// * `poissons_ratio` - Poisson's ratio.
    /// * `density` - Density (kg/m^3).
    /// * `friction_coefficient` - Coefficient of friction.
    /// * `restitution_coefficient` - Coefficient of restitution (0.0 to 1.0).
    /// * `surface_energy` - Surface energy (J/m^2), used for JKR/sJKR models.
    fn set_particle_material(&mut self, youngs_modulus: f32, poissons_ratio: f32, density: f32, friction_coefficient: f32, restitution_coefficient: f32, surface_energy: f32) {
        self.inner.particle_material = material::Material::new(youngs_modulus, poissons_ratio, density, friction_coefficient, restitution_coefficient, surface_energy);
        // Also update existing particles? 
        // For now, density is used for mass calculation during creation, but mass is stored on particle.
        // If we change density, we might want to update mass? 
        // Or just assume this is called before adding particles.
        // But mass is passed in add_particle.
        // The user asked to specify density. 
        // If add_particle takes mass, density is redundant or used for calculating mass from radius.
        // Let's assume add_particle takes mass, so density in material is just for reference or future use (e.g. if we add by radius only).
        // However, effective mass calculation in physics uses particle mass.
        // So density in Material struct is maybe not used if we use particle.mass?
        // Wait, effective_mass uses m1, m2.
        // So density in Material is unused?
        // Let's keep it consistent.
    }

    #[pyo3(signature = (youngs_modulus, poissons_ratio, density, friction_coefficient, restitution_coefficient, surface_energy=0.0))]
    fn set_wall_material(&mut self, youngs_modulus: f32, poissons_ratio: f32, density: f32, friction_coefficient: f32, restitution_coefficient: f32, surface_energy: f32) {
        self.inner.wall_material = material::Material::new(youngs_modulus, poissons_ratio, density, friction_coefficient, restitution_coefficient, surface_energy);
    }

    /// Enables or disables periodic boundary conditions.
    ///
    /// # Arguments
    ///
    /// * `x` - Enable periodicity in X direction.
    /// * `y` - Enable periodicity in Y direction.
    /// * `z` - Enable periodicity in Z direction.
    fn set_periodic_boundary(&mut self, x: bool, y: bool, z: bool) {
        self.inner.periodic = [x, y, z];
    }

    /// Enables GPU acceleration for the simulation.
    ///
    /// This method initializes the GPU resources and transfers particle data to the GPU.
    /// Requires a compatible GPU (Metal on MacOS, Vulkan/DX12 on Windows/Linux).
    fn enable_gpu(&mut self) -> PyResult<()> {
        self.inner.enable_gpu().map_err(|e| PyRuntimeError::new_err(e))
    }

    fn init_mpi(&mut self) {
        self.inner.init_mpi();
    }

    /// Sets the contact force models.
    ///
    /// # Arguments
    ///
    /// * `normal_model` - The normal force model ("linear", "hertzian", "jkr", "sjkr").
    /// * `tangential_model` - The tangential force model ("coulomb").
    fn set_contact_models(&mut self, normal: String, tangential: String) -> PyResult<()> {
        let normal_model = match normal.to_lowercase().as_str() {
            "hertz" | "hertzian" => physics::NormalForceModel::Hertzian,
            "linear" | "spring_dashpot" => physics::NormalForceModel::LinearSpringDashpot,
            "hysteretic" => physics::NormalForceModel::Hysteretic,
            "jkr" => physics::NormalForceModel::JKR,
            "sjkr" | "simplified_jkr" => physics::NormalForceModel::SimplifiedJKR,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid normal force model")),
        };

        let tangential_model = match tangential.to_lowercase().as_str() {
            "mindlin" => physics::TangentialForceModel::Mindlin,
            "coulomb" => physics::TangentialForceModel::Coulomb,
            "linear" | "spring_coulomb" => physics::TangentialForceModel::LinearSpringCoulomb,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid tangential force model")),
        };

        self.inner.normal_model = normal_model;
        self.inner.tangential_model = tangential_model;
        Ok(())
    }

    fn run(&mut self, duration: f32) {
        let steps = (duration / self.inner.dt).ceil() as usize;
        let pb = indicatif::ProgressBar::new(steps as u64);
        pb.set_style(indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
            
        for _ in 0..steps {
            self.inner.step();
            pb.inc(1);
        }
        pb.finish_with_message("Simulation complete");
    }

    /// Writes the current simulation state to a VTK file.
    ///
    /// # Arguments
    ///
    /// * `filename` - The path to the output file.
    fn write_vtk(&mut self, filename: String) -> PyResult<()> {
        // Sync before writing
        self.inner.sync_particles();
        vtk::write_vtk(&filename, &self.inner.particles).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(())
    }
    
    /// Returns the total number of particles in the simulation.
    fn get_particle_count(&self) -> usize {
        self.inner.particles.len()
    }

    /// Retrieves the position of a particle.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the particle.
    ///
    /// # Returns
    ///
    /// A tuple `(x, y, z)` containing the particle's position.
    fn get_particle_position(&self, id: usize) -> PyResult<(f32, f32, f32)> {
        if id >= self.inner.particles.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Particle index out of bounds"));
        }
        let p = &self.inner.particles[id];
        Ok((p.position.x, p.position.y, p.position.z))
    }

    /// Retrieves the velocity of a particle.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the particle.
    ///
    /// # Returns
    ///
    /// A tuple `(vx, vy, vz)` containing the particle's velocity.
    fn get_particle_velocity(&self, id: usize) -> PyResult<(f32, f32, f32)> {
        if id >= self.inner.particles.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Particle index out of bounds"));
        }
        let p = &self.inner.particles[id];
        Ok((p.velocity.x, p.velocity.y, p.velocity.z))
    }

    fn get_particle_angular_velocity(&self, index: usize) -> PyResult<(f32, f32, f32)> {
        if index >= self.inner.particles.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Particle index out of bounds"));
        }
        let p = &self.inner.particles[index];
        Ok((p.angular_velocity.x, p.angular_velocity.y, p.angular_velocity.z))
    }

    /// Sets the velocity of a particle.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the particle.
    /// * `vx`, `vy`, `vz` - The new velocity components.
    fn set_particle_velocity(&mut self, id: usize, vx: f32, vy: f32, vz: f32) -> PyResult<()> {
        if id >= self.inner.particles.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Particle index out of bounds"));
        }
        self.inner.particles[id].velocity = Vec3::new(vx, vy, vz);
        Ok(())
    }

    fn set_particle_angular_velocity(&mut self, index: usize, x: f32, y: f32, z: f32) -> PyResult<()> {
        if index >= self.inner.particles.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Particle index out of bounds"));
        }
        self.inner.particles[index].angular_velocity = Vec3::new(x, y, z);
        Ok(())
    }
}
