pub mod particle;
pub mod grid;
pub mod simulation;
pub mod vtk;
pub mod mesh;

use pyo3::prelude::*;
use glam::Vec3;

#[pymodule]
fn rusty_particles(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySimulation>()?;
    
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

    fn write_vtk(&self, filename: String) -> PyResult<()> {
        vtk::write_vtk(&filename, &self.inner.particles).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn particle_count(&self) -> usize {
        self.inner.particles.len()
    }
}
