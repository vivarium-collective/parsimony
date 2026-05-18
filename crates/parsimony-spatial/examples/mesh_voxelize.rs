//! `mesh_voxelize` — voxelize a triangle mesh into a `VoxelField` and
//! print stats. Demonstrates the in/out classification pipeline that
//! the placement loop will use for compartment construction.
//!
//! The mesh is a procedurally-tessellated sphere via parry3d's
//! `Ball::to_trimesh`. After voxelization we run `prune()` to
//! collapse uniform interior 8³ tiles into `L1Slot::Tile` values —
//! the same compaction we documented in §6.4 of the design doc.
//!
//! For the eventual cellPACK comparison harness, this is the skeleton
//! to extend: replace the `Ball::to_trimesh` call with an .obj loader
//! that reads `cellpack/examples/recipes/v2/peroxisome.json`'s
//! membrane mesh, transform into the recipe's bounding box, run the
//! same voxelize, then compare classifications against cellPACK's
//! Python output (trimesh.voxel.creation.voxelize + flood-fill).
//!
//! Run with: `cargo run --release --example mesh_voxelize -p parsimony-spatial`

use std::time::Instant;

use parsimony_spatial::{
    prepare_trimesh_for_voxelize, voxelize_trimesh, Aabb, Cell, VoxelField,
};

fn aabb_cube(half: f32) -> Aabb {
    use nalgebra::Point3;
    Aabb::new(Point3::new(-half, -half, -half), Point3::new(half, half, half))
}

fn main() {
    let radius = 30.0_f32;
    let cell_size = 1.0_f32;
    let bounds = aabb_cube(radius + 2.0);

    // Build a tessellated sphere and prepare it for in/out queries.
    let ball = parry3d::shape::Ball::new(radius);
    let (vertices, indices) = ball.to_trimesh(40, 40);
    let mut mesh = parry3d::shape::TriMesh::new(vertices, indices).expect("trimesh");
    let n_tris = mesh.indices().len();
    prepare_trimesh_for_voxelize(&mut mesh).expect("orient");

    println!(
        "Mesh: parry3d Ball(r={radius}) tessellated to {} triangles",
        n_tris
    );
    println!(
        "VoxelField: cell_size={cell_size}, bounds=[{:.1}..{:.1}]³",
        bounds.min.x, bounds.max.x
    );
    println!();

    let mut field = VoxelField::new(cell_size);
    let t = Instant::now();
    voxelize_trimesh(&mut field, &mesh, 1, bounds);
    let voxel_time = t.elapsed();

    let before = field.stats();
    println!("After voxelize ({:.2?}):", voxel_time);
    print_stats("  ", &before);

    let t = Instant::now();
    field.prune();
    let prune_time = t.elapsed();
    let after = field.stats();

    println!("\nAfter prune ({:.2?}):", prune_time);
    print_stats("  ", &after);

    println!("\nDifference:");
    println!(
        "  dense L0 tiles: {} → {}  ({} collapsed to uniform)",
        before.dense_l0_tiles,
        after.dense_l0_tiles,
        before.dense_l0_tiles.saturating_sub(after.dense_l0_tiles)
    );
    println!(
        "  L1 slot tiles (non-bg): {} → {}",
        before.tile_l1_slots, after.tile_l1_slots
    );
    println!(
        "  memory: {} → {} bytes  ({:.1}% saved)",
        before.memory_bytes,
        after.memory_bytes,
        100.0 * (1.0 - after.memory_bytes as f64 / before.memory_bytes as f64)
    );

    // Sanity checks.
    use nalgebra::Point3;
    let centre = field.sample(Point3::origin());
    println!(
        "\nSanity: sample(origin).compartment = {} (expected 1)",
        centre.compartment
    );
    let outside = field.sample(Point3::new(radius + 1.0, 0.0, 0.0));
    println!(
        "Sanity: sample(outside).compartment = {} (expected 0)",
        outside.compartment
    );
    assert_eq!(centre, Cell::new(1, 0, 0), "centre should be interior");
    assert_eq!(outside.compartment, 0, "outside should be default");

    // Volume check.
    let analytical_volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
    let voxel_volume = after.active_cells as f32 * cell_size.powi(3);
    let pct = 100.0 * voxel_volume / analytical_volume;
    println!(
        "\nVolume: {:.0} analytical, {:.0} voxel ({:.1}% — discretization error)",
        analytical_volume, voxel_volume, pct
    );
}

fn print_stats(prefix: &str, s: &parsimony_spatial::VoxelFieldStats) {
    println!("{prefix}active cells:    {}", s.active_cells);
    println!("{prefix}dense L0 tiles:  {}", s.dense_l0_tiles);
    println!("{prefix}const L0 tiles:  {}", s.constant_l0_tiles);
    println!("{prefix}L1 slot tiles:   {}", s.tile_l1_slots);
    println!("{prefix}L1 tiles:        {}", s.l1_tiles);
    println!("{prefix}const root tiles:{}", s.constant_root_tiles);
    println!("{prefix}memory bytes:    {}", s.memory_bytes);
}
