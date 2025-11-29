use glam::Vec3;
use std::fs::File;
use std::io::BufReader;

#[derive(Clone, Debug)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub normal: Vec3,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, normal: Vec3) -> Self {
        Self { v0, v1, v2, normal }
    }

    // Check collision with a sphere
    // Returns (penetration_depth, normal) if collision occurs
    pub fn intersect_sphere(&self, center: Vec3, radius: f32) -> Option<(f32, Vec3)> {
        // 1. Find closest point on triangle to sphere center
        let closest = self.closest_point(center);
        
        // 2. Check distance
        let dist_vec = center - closest;
        let dist_sq = dist_vec.length_squared();
        
        if dist_sq < radius * radius {
            let dist = dist_sq.sqrt();
            let penetration = radius - dist;
            
            // Normal points from triangle to sphere
            let normal = if dist > 1e-6 {
                dist_vec / dist
            } else {
                self.normal // Fallback if center is exactly on triangle
            };
            
            Some((penetration, normal))
        } else {
            None
        }
    }

    fn closest_point(&self, p: Vec3) -> Vec3 {
        let edge0 = self.v1 - self.v0;
        let edge1 = self.v2 - self.v0;
        let v0 = self.v0 - p;

        let a = edge0.dot(edge0);
        let b = edge0.dot(edge1);
        let c = edge1.dot(edge1);
        let d = edge0.dot(v0);
        let e = edge1.dot(v0);

        let det = a * c - b * b;
        let s = b * e - c * d;
        let t = b * d - a * e;

        let mut s_clamped = s;
        let mut t_clamped = t;

        if s + t < det {
            if s < 0.0 {
                if t < 0.0 {
                    if d < 0.0 {
                        s_clamped = (-d).clamp(0.0, a);
                        t_clamped = 0.0;
                    } else {
                        s_clamped = 0.0;
                        t_clamped = (-e).clamp(0.0, c);
                    }
                } else {
                    s_clamped = 0.0;
                    t_clamped = (-e).clamp(0.0, c);
                }
            } else if t < 0.0 {
                s_clamped = (-d).clamp(0.0, a);
                t_clamped = 0.0;
            } else {
                let inv_det = 1.0 / det;
                s_clamped *= inv_det;
                t_clamped *= inv_det;
            }
        } else {
            if s < 0.0 {
                let tmp0 = b + d;
                let tmp1 = c + e;
                if tmp1 > tmp0 {
                    let numer = tmp1 - tmp0;
                    let denom = a - 2.0 * b + c;
                    s_clamped = (numer / denom).clamp(0.0, 1.0);
                    t_clamped = 1.0 - s_clamped;
                } else {
                    t_clamped = (-e).clamp(0.0, c);
                    s_clamped = 0.0;
                }
            } else if t < 0.0 {
                if a + d > b + e {
                    let numer = c + e - b - d;
                    let denom = a - 2.0 * b + c;
                    s_clamped = (numer / denom).clamp(0.0, 1.0);
                    t_clamped = 1.0 - s_clamped;
                } else {
                    s_clamped = (-e).clamp(0.0, c);
                    t_clamped = 0.0;
                }
            } else {
                let numer = c + e - b - d;
                let denom = a - 2.0 * b + c;
                s_clamped = (numer / denom).clamp(0.0, 1.0);
                t_clamped = 1.0 - s_clamped;
            }
        }

        self.v0 + s_clamped * edge0 + t_clamped * edge1
    }
}

pub struct Mesh {
    pub triangles: Vec<Triangle>,
}

impl Mesh {
    pub fn load_stl(path: &str) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let stl = stl_io::read_stl(&mut file)?;
        
        let triangles = stl.faces.iter().map(|face| {
            let v0_raw = stl.vertices[face.vertices[0]];
            let v1_raw = stl.vertices[face.vertices[1]];
            let v2_raw = stl.vertices[face.vertices[2]];

            let v0 = Vec3::new(v0_raw[0], v0_raw[1], v0_raw[2]);
            let v1 = Vec3::new(v1_raw[0], v1_raw[1], v1_raw[2]);
            let v2 = Vec3::new(v2_raw[0], v2_raw[1], v2_raw[2]);
            
            let normal = Vec3::new(face.normal[0], face.normal[1], face.normal[2]);
            Triangle::new(v0, v1, v2, normal)
        }).collect();

        Ok(Self { triangles })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_intersection() {
        let t = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        // Sphere directly above triangle
        let sphere_pos = Vec3::new(0.2, 0.2, 0.5);
        let radius = 0.6; // Should intersect (dist is 0.5)
        
        let result = t.intersect_sphere(sphere_pos, radius);
        assert!(result.is_some());
        let (penetration, normal) = result.unwrap();
        assert!((penetration - 0.1).abs() < 1e-5);
        assert!((normal - Vec3::Z).length() < 1e-5);

        // Sphere far away
        let result = t.intersect_sphere(Vec3::new(0.2, 0.2, 2.0), 0.5);
        assert!(result.is_none());
    }
}
