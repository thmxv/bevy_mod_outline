use std::{cmp::Ordering, mem};

use bevy::{
    math::FloatOrd,
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_resource::{PrimitiveTopology, VertexFormat},
    },
    utils::{HashMap, HashSet},
};
use itertools::Itertools;

use crate::ATTRIBUTE_OUTLINE_NORMAL;

enum IndexIterator<'a> {
    ExplicitU16(std::slice::Iter<'a, u16>),
    ExplicitU32(std::slice::Iter<'a, u32>),
    Implicit(std::ops::Range<usize>),
}

impl<'a> From<&'a Mesh> for IndexIterator<'a> {
    fn from(value: &'a Mesh) -> Self {
        match value.indices() {
            Some(Indices::U16(vec)) => IndexIterator::ExplicitU16(vec.iter()),
            Some(Indices::U32(vec)) => IndexIterator::ExplicitU32(vec.iter()),
            None => IndexIterator::Implicit(0..value.count_vertices()),
        }
    }
}

impl Iterator for IndexIterator<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IndexIterator::ExplicitU16(iter) => iter.next().map(|val| *val as usize),
            IndexIterator::ExplicitU32(iter) => iter.next().map(|val| *val as usize),
            IndexIterator::Implicit(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IndexIterator::ExplicitU16(iter) => iter.size_hint(),
            IndexIterator::ExplicitU32(iter) => iter.size_hint(),
            IndexIterator::Implicit(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for IndexIterator<'_> {}

/// Failed to generate outline normals for the mesh.
#[derive(thiserror::Error, Debug)]
pub enum GenerateOutlineNormalsError {
    #[error("unsupported primitive topology '{0:?}'")]
    UnsupportedPrimitiveTopology(PrimitiveTopology),
    #[error("missing vertex attributes '{0}'")]
    MissingVertexAttribute(&'static str),
    #[error("the '{0}' vertex attribute should have {1:?} format, but had {2:?} format")]
    InvalidVertexAttributeFormat(&'static str, VertexFormat, VertexFormat),
}

/// Extension methods for [`Mesh`].
pub trait OutlineMeshExt {
    /// Generates outline normals for the mesh.
    ///
    /// Vertex extrusion only works for meshes with smooth surface normals. Hard edges cause
    /// visual artefacts. This function generates faux-smooth normals for outlining purposes
    /// by grouping vertices by their position and averaging the normals at each point. These
    /// outline normals are then inserted as a separate vertex attribute so that the regular
    /// normals remain untouched. However, insofar as the outline normals are not
    /// perpendicular to the surface of the mesh, this technique may result in non-uniform
    /// outline thickness.
    ///
    /// This function only supports meshes with TriangleList topology.
    fn generate_outline_normals(&mut self) -> Result<(), GenerateOutlineNormalsError>;
    fn modify_for_non_manifold_outlines(&mut self) -> Result<(), GenerateOutlineNormalsError>;
}

impl OutlineMeshExt for Mesh {
    fn generate_outline_normals(&mut self) -> Result<(), GenerateOutlineNormalsError> {
        if self.primitive_topology() != PrimitiveTopology::TriangleList {
            return Err(GenerateOutlineNormalsError::UnsupportedPrimitiveTopology(
                self.primitive_topology(),
            ));
        }
        let positions = match self.attribute(Mesh::ATTRIBUTE_POSITION).ok_or(
            GenerateOutlineNormalsError::MissingVertexAttribute(Mesh::ATTRIBUTE_POSITION.name),
        )? {
            VertexAttributeValues::Float32x3(p) => Ok(p),
            v => Err(GenerateOutlineNormalsError::InvalidVertexAttributeFormat(
                Mesh::ATTRIBUTE_POSITION.name,
                VertexFormat::Float32x3,
                v.into(),
            )),
        }?;
        let normals = match self.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(p)) => Some(p),
            _ => None,
        };
        let mut map = HashMap::<[FloatOrd; 3], Vec3>::with_capacity(positions.len());
        let mut it = IndexIterator::from(&*self);
        while let (Some(i0), Some(i1), Some(i2)) = (it.next(), it.next(), it.next()) {
            for (j0, j1, j2) in [(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
                let p0 = Vec3::from(positions[j0]);
                let p1 = Vec3::from(positions[j1]);
                let p2 = Vec3::from(positions[j2]);
                let angle = (p1 - p0).angle_between(p2 - p0);
                let n = map
                    .entry([FloatOrd(p0.x), FloatOrd(p0.y), FloatOrd(p0.z)])
                    .or_default();
                *n += angle
                    * if let Some(ns) = normals {
                        // Use vertex normal
                        Vec3::from(ns[j0])
                    } else {
                        // Calculate face normal
                        (p1 - p0).cross(p2 - p0).normalize_or_zero()
                    };
            }
        }
        let mut outlines = Vec::with_capacity(positions.len());
        for p in positions.iter() {
            let key = [FloatOrd(p[0]), FloatOrd(p[1]), FloatOrd(p[2])];
            outlines.push(
                map.get(&key)
                    .copied()
                    .unwrap_or(Vec3::ZERO)
                    .normalize_or_zero()
                    .to_array(),
            );
        }
        self.insert_attribute(
            ATTRIBUTE_OUTLINE_NORMAL,
            VertexAttributeValues::Float32x3(outlines),
        );
        Ok(())
    }

    fn modify_for_non_manifold_outlines(&mut self) -> Result<(), GenerateOutlineNormalsError> {
        // TODO: Maybe make optional
        self.generate_outline_normals()?;

        if self.primitive_topology() != PrimitiveTopology::TriangleList {
            return Err(GenerateOutlineNormalsError::UnsupportedPrimitiveTopology(
                self.primitive_topology(),
            ));
        }
        let positions = match self.attribute(Mesh::ATTRIBUTE_POSITION).ok_or(
            GenerateOutlineNormalsError::MissingVertexAttribute(Mesh::ATTRIBUTE_POSITION.name),
        )? {
            VertexAttributeValues::Float32x3(p) => Ok(p),
            v => Err(GenerateOutlineNormalsError::InvalidVertexAttributeFormat(
                Mesh::ATTRIBUTE_POSITION.name,
                VertexFormat::Float32x3,
                v.into(),
            )),
        }?;
        let mut it = IndexIterator::from(&*self);
        let mut edges_map =
            HashMap::<([FloatOrd; 3], [FloatOrd; 3]), Vec<(usize, usize, usize)>>::with_capacity(
                it.len(),
            );
        // For each face
        while let (Some(i0), Some(i1), Some(i2)) = (it.next(), it.next(), it.next()) {
            // For each edge in face
            for (j0, j1, j2) in [(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
                let mut p0 = Vec3::from(positions[j0]);
                let mut p1 = Vec3::from(positions[j1]);
                // Sort to make sure edges with same 2 vertices have the same key
                // TODO: uggly, ineficient and possibly not (always) correct (NaN?)
                let cmp = {
                    let xcmp = p0.x.total_cmp(&p1.x);
                    if xcmp != Ordering::Equal {
                        xcmp
                    } else {
                        let ycmp = p0.y.total_cmp(&p1.y);
                        if ycmp != Ordering::Equal {
                            ycmp
                        } else {
                            p0.z.total_cmp(&p1.z)
                        }
                    }
                };
                if cmp == Ordering::Greater {
                    mem::swap(&mut p0, &mut p1);
                };
                let key0 = [FloatOrd(p0.x), FloatOrd(p0.y), FloatOrd(p0.z)];
                let key1 = [FloatOrd(p1.x), FloatOrd(p1.y), FloatOrd(p1.z)];
                // TODO: Use small vector for optimization
                let edge_faces = edges_map.entry((key0, key1)).or_insert(vec![]);
                edge_faces.push((j0, j1, j2));
            }
        }
        let mut new_quad_faces = Vec::<(usize, usize, Vec3)>::new();
        for edge_faces in edges_map.values() {
            if edge_faces.len() == 1 {
                let (j0, j1, j2) = edge_faces[0];
                let p0 = Vec3::from(positions[j0]);
                let p1 = Vec3::from(positions[j1]);
                let p2 = Vec3::from(positions[j2]);
                let face_normal = (p1 - p0).cross(p2 - p0);
                let edge_direction = p1 - p0;
                let edge_normal = edge_direction.cross(face_normal).normalize_or_zero();
                new_quad_faces.push((j0, j1, edge_normal));
            }
        }
        let mut extrude_verts_map = HashMap::<usize, Vec<usize>>::new();
        for new_quad_face in new_quad_faces {
            let (i0, i1, normal) = new_quad_face;
            let (i2, i3) = add_quad_face_indexed(self, i0, i1, normal);
            let extruded_vec = extrude_verts_map.entry(i0).or_default();
            extruded_vec.push(i2);
            let extruded_vec = extrude_verts_map.entry(i1).or_default();
            extruded_vec.push(i3);
        }
        // TODO: Maybe face with other indices order
        for (i0, extruded) in extrude_verts_map {
            for (i1, i2) in extruded.iter().copied().tuple_windows() {
                add_tri_face(self, i0, i1, i2);
            }
        }
        Ok(())
    }
}

fn add_tri_face(mesh: &mut Mesh, i0: usize, i1: usize, i2: usize) {
    let indices = mesh.indices_mut();
    match indices {
        Some(Indices::U16(indices)) => {
            indices.push(i0 as u16);
            indices.push(i1 as u16);
            indices.push(i2 as u16);
        }
        Some(Indices::U32(indices)) => {
            indices.push(i0 as u32);
            indices.push(i1 as u32);
            indices.push(i2 as u32);
        }
        None => panic!(),
    }
}

fn add_quad_face_indexed(mesh: &mut Mesh, i0: usize, i1: usize, normal: Vec3) -> (usize, usize) {
    let VertexAttributeValues::Float32x3(positions) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION).unwrap()
    else {
        panic!();
    };
    let i2 = positions.len();
    let p0 = positions[i0];
    positions.push(p0);
    let i3 = positions.len();
    let p1 = positions[i1];
    positions.push(p1);

    let VertexAttributeValues::Float32x3(outline_normals) =
        mesh.attribute_mut(ATTRIBUTE_OUTLINE_NORMAL).unwrap()
    else {
        panic!();
    };
    outline_normals.push(normal.to_array());
    outline_normals.push(normal.to_array());

    add_tri_face(mesh, i1, i0, i3);
    add_tri_face(mesh, i0, i2, i3);

    for (attribute_id, attribute) in mesh.attributes_mut() {
        if attribute_id != Mesh::ATTRIBUTE_POSITION.id
            && attribute_id != ATTRIBUTE_OUTLINE_NORMAL.id
        {
            match attribute {
                VertexAttributeValues::Float32(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Sint32(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Uint32(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Float32x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Sint32x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Uint32x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Float32x3(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Sint32x3(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Uint32x3(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Float32x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Sint32x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Uint32x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Sint16x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Snorm16x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Uint16x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Unorm16x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Sint16x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Snorm16x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Uint16x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Unorm16x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Sint8x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Snorm8x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Uint8x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Unorm8x2(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Sint8x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Snorm8x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Uint8x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
                VertexAttributeValues::Unorm8x4(attribute) => {
                    let a0 = attribute[i0];
                    attribute.push(a0);
                    let a1 = attribute[i1];
                    attribute.push(a1);
                }
            }
        }
    }
    (i2, i3)
}

fn auto_generate_outline_normals(
    mut meshes: ResMut<Assets<Mesh>>,
    mut events: EventReader<'_, '_, AssetEvent<Mesh>>,
    mut squelch: Local<HashSet<AssetId<Mesh>>>,
) {
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                if squelch.contains(id) {
                    // Suppress modification events created by this system
                    squelch.remove(id);
                } else if let Some(mesh) = meshes.get_mut(*id) {
                    let _ = mesh.generate_outline_normals();
                    squelch.insert(*id);
                }
            }
            AssetEvent::Removed { id } => {
                squelch.remove(id);
            }
            _ => {}
        }
    }
}

fn auto_generate_non_manifold_outline_normals(
    mut meshes: ResMut<Assets<Mesh>>,
    mut events: EventReader<'_, '_, AssetEvent<Mesh>>,
    mut squelch: Local<HashSet<AssetId<Mesh>>>,
) {
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                if squelch.contains(id) {
                    // Suppress modification events created by this system
                    squelch.remove(id);
                } else if let Some(mesh) = meshes.get_mut(*id) {
                    let _ = mesh.modify_for_non_manifold_outlines();
                    squelch.insert(*id);
                }
            }
            AssetEvent::Removed { id } => {
                squelch.remove(id);
            }
            _ => {}
        }
    }
}

#[derive(Default)]
pub enum NormalGenerationMode {
    #[default]
    Default,
    NonManifold,
}

/// Automatically runs [`generate_outline_normals`](OutlineMeshExt::generate_outline_normals)
/// on every mesh.
///
/// This is provided as a convenience for simple projects. It runs the outline normal
/// generator every time a mesh asset is created or modified without consideration for
/// whether this is necessary or appropriate.
#[derive(Default)]
pub struct AutoGenerateOutlineNormalsPlugin {
    pub generation_mode: NormalGenerationMode,
}

impl Plugin for AutoGenerateOutlineNormalsPlugin {
    fn build(&self, app: &mut App) {
        match self.generation_mode {
            NormalGenerationMode::Default => {
                app.add_systems(Update, auto_generate_outline_normals);
            }
            NormalGenerationMode::NonManifold => {
                app.add_systems(Update, auto_generate_non_manifold_outline_normals);
            }
        }
    }
}
