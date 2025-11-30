use rusty_particles::simulation::Simulation;
use rusty_particles::vtk::write_vtk;
use rusty_particles::mesh::Mesh;
use glam::Vec3;
use indicatif::ProgressBar;
use std::fs;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to STL file for walls
    #[arg(short, long)]
    stl: Option<String>,

    /// Number of particles
    #[arg(short, long, default_value_t = 5000)]
    count: usize,
}

fn main() {
    let args = Args::parse();

    // Parameters
    let dt = 0.001;
    let duration = 2.0;
    let steps = (duration / dt) as usize;
    let output_interval = 10; 
    
    let bounds_min = Vec3::new(-1.0, 0.0, -1.0);
    let bounds_max = Vec3::new(1.0, 5.0, 1.0);
    
    println!("Initializing simulation with {} particles...", args.count);
    let mut sim = Simulation::new(dt, bounds_min, bounds_max, args.count);

    // Load STL if provided
    if let Some(stl_path) = args.stl {
        println!("Loading STL from: {}", stl_path);
        match Mesh::load_stl(&stl_path) {
            Ok(mesh) => {
                println!("Loaded mesh with {} triangles", mesh.triangles.len());
                sim.add_mesh(mesh);
            },
            Err(e) => eprintln!("Error loading STL: {}", e),
        }
    }

    // Initialize particles: Drop them in a column
    let mut rng = fastrand::Rng::new();
    for _ in 0..args.count {
        let x = rng.f32() * 0.8 - 0.4;
        let z = rng.f32() * 0.8 - 0.4;
        let y = rng.f32() * 3.0 + 1.0;
        let radius = 0.02 + rng.f32() * 0.01; 
        let mass = 1.0; 
        sim.add_particle(Vec3::new(x, y, z), radius, mass, false);
    }

    // Create output directory
    fs::create_dir_all("output").unwrap();

    println!("Starting simulation: {} steps", steps);
    let bar = ProgressBar::new(steps as u64);

    for step in 0..steps {
        sim.step();
        
        if step % output_interval == 0 {
            let filename = format!("output/step_{:05}.vtk", step);
            if let Err(e) = write_vtk(&filename, &sim.particles) {
                eprintln!("Error writing VTK: {}", e);
            }
        }
        bar.inc(1);
    }
    bar.finish();
    println!("Simulation complete. Remaining particles: {}", sim.particles.len());
}
