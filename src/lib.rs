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
use rayon::ThreadPoolBuilder;

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

#[pyclass(name = "Simulation")]
struct PySimulation {
    inner: Box<simulation::Simulation>,
}

#[pymethods]
impl PySimulation {
    #[staticmethod]
    fn create(dt: f32, min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32, particle_count: usize) -> Self {
        let min = Vec3::new(min_x, min_y, min_z);
        let max = Vec3::new(max_x, max_y, max_z);
        
        let sim = simulation::Simulation::new(dt, min, max, particle_count);
        
        Self {
            inner: Box::new(sim),
        }
    }

    fn add_particle(&mut self, x: f32, y: f32, z: f32, radius: f32, mass: f32) {
        let pos = Vec3::new(x, y, z);
        self.inner.add_particle(pos, radius, mass);
    }

    fn add_mesh(&mut self, path: String) -> PyResult<()> {
        let mesh = mesh::Mesh::load_stl(&path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        self.inner.add_mesh(mesh);
        Ok(())
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn set_particle_material(&mut self, youngs_modulus: f32, poissons_ratio: f32, density: f32, friction_coefficient: f32, restitution_coefficient: f32) {
        self.inner.particle_material = material::Material::new(youngs_modulus, poissons_ratio, density, friction_coefficient, restitution_coefficient);
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

    fn set_wall_material(&mut self, youngs_modulus: f32, poissons_ratio: f32, density: f32, friction_coefficient: f32, restitution_coefficient: f32) {
        self.inner.wall_material = material::Material::new(youngs_modulus, poissons_ratio, density, friction_coefficient, restitution_coefficient);
    }

    fn set_periodic(&mut self, x: bool, y: bool, z: bool) {
        self.inner.periodic = [x, y, z];
    }

    fn enable_gpu(&mut self) {
        self.inner.enable_gpu();
    }

    fn init_mpi(&mut self) {
        self.inner.init_mpi();
    }

    fn set_contact_models(&mut self, normal: String, tangential: String) -> PyResult<()> {
        let normal_model = match normal.to_lowercase().as_str() {
            "hertz" | "hertzian" => physics::NormalForceModel::Hertzian,
            "linear" | "spring_dashpot" => physics::NormalForceModel::LinearSpringDashpot,
            "hysteretic" => physics::NormalForceModel::Hysteretic,
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

    fn write_vtk(&mut self, filename: String) -> PyResult<()> {
        // Sync before writing
        self.inner.sync_particles();
        vtk::write_vtk(&filename, &self.inner.particles).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn particle_count(&self) -> usize {
        self.inner.particles.len()
    }
}
