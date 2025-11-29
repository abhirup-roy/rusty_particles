use std::fs::File;
use std::io::{Write, BufWriter};
use crate::particle::Particle;

pub fn write_vtk(filename: &str, particles: &[Particle]) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# vtk DataFile Version 3.0")?;
    writeln!(writer, "DEM Simulation Data")?;
    writeln!(writer, "ASCII")?;
    writeln!(writer, "DATASET POLYDATA")?;
    
    // Points
    writeln!(writer, "POINTS {} float", particles.len())?;
    for p in particles {
        writeln!(writer, "{} {} {}", p.position.x, p.position.y, p.position.z)?;
    }

    // Point Data (Attributes)
    writeln!(writer, "POINT_DATA {}", particles.len())?;
    
    // Velocities
    writeln!(writer, "VECTORS velocity float")?;
    for p in particles {
        writeln!(writer, "{} {} {}", p.velocity.x, p.velocity.y, p.velocity.z)?;
    }

    // Radii (Scalars)
    writeln!(writer, "SCALARS radius float 1")?;
    writeln!(writer, "LOOKUP_TABLE default")?;
    for p in particles {
        writeln!(writer, "{}", p.radius)?;
    }
    
    // IDs (Scalars)
    writeln!(writer, "SCALARS id int 1")?;
    writeln!(writer, "LOOKUP_TABLE default")?;
    for p in particles {
        writeln!(writer, "{}", p.id)?;
    }

    Ok(())
}
