use glam::{Vec3, IVec3};
use crate::particle::Particle;

pub struct Grid {
    cell_size: f32,
    cells: Vec<Vec<usize>>, // Stores particle indices
    grid_origin: Vec3,
    grid_dims: IVec3,
}

impl Grid {
    pub fn new(cell_size: f32, min_bound: Vec3, max_bound: Vec3) -> Self {
        let grid_dims = ((max_bound - min_bound) / cell_size).ceil().as_ivec3() + IVec3::ONE;
        let total_cells = (grid_dims.x * grid_dims.y * grid_dims.z) as usize;
        
        Self {
            cell_size,
            cells: vec![Vec::new(); total_cells],
            grid_origin: min_bound,
            grid_dims,
        }
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    fn get_cell_index(&self, position: Vec3) -> Option<usize> {
        let relative_pos = position - self.grid_origin;
        let coords = (relative_pos / self.cell_size).floor().as_ivec3();

        if coords.x < 0 || coords.y < 0 || coords.z < 0 ||
           coords.x >= self.grid_dims.x || coords.y >= self.grid_dims.y || coords.z >= self.grid_dims.z {
            return None;
        }

        Some((coords.x + coords.y * self.grid_dims.x + coords.z * self.grid_dims.x * self.grid_dims.y) as usize)
    }

    pub fn insert(&mut self, particle: &Particle) {
        if let Some(idx) = self.get_cell_index(particle.position) {
            self.cells[idx].push(particle.id);
        }
    }

    pub fn get_potential_collisions(&self, particle: &Particle, periodic: [bool; 3]) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let relative_pos = particle.position - self.grid_origin;
        let center_coords = (relative_pos / self.cell_size).floor().as_ivec3();

        // Check 3x3x3 neighborhood
        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    let mut neighbor_coords = center_coords + IVec3::new(x, y, z);
                    
                    // Handle periodicity
                    if periodic[0] {
                        if neighbor_coords.x < 0 { neighbor_coords.x += self.grid_dims.x; }
                        else if neighbor_coords.x >= self.grid_dims.x { neighbor_coords.x -= self.grid_dims.x; }
                    }
                    if periodic[1] {
                        if neighbor_coords.y < 0 { neighbor_coords.y += self.grid_dims.y; }
                        else if neighbor_coords.y >= self.grid_dims.y { neighbor_coords.y -= self.grid_dims.y; }
                    }
                    if periodic[2] {
                        if neighbor_coords.z < 0 { neighbor_coords.z += self.grid_dims.z; }
                        else if neighbor_coords.z >= self.grid_dims.z { neighbor_coords.z -= self.grid_dims.z; }
                    }
                    
                    if neighbor_coords.x >= 0 && neighbor_coords.x < self.grid_dims.x &&
                       neighbor_coords.y >= 0 && neighbor_coords.y < self.grid_dims.y &&
                       neighbor_coords.z >= 0 && neighbor_coords.z < self.grid_dims.z {
                        
                        let idx = (neighbor_coords.x + 
                                   neighbor_coords.y * self.grid_dims.x + 
                                   neighbor_coords.z * self.grid_dims.x * self.grid_dims.y) as usize;
                        
                        neighbors.extend_from_slice(&self.cells[idx]);
                    }
                }
            }
        }
        neighbors
    }
}
