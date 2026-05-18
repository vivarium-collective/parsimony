//! Criterion benchmarks comparing the three [`SpatialIndex`] impls —
//! [`BruteIndex`], [`BvhIndex`] (binary), [`QbvhIndex`] (4-wide SIMD) —
//! across four workloads:
//!
//! 1. `bulk_build` — build a tree from `n` `(uid, aabb)` pairs.
//! 2. `incremental_insert` — insert `n` items one at a time.
//! 3. `query_*` — AABB and sphere queries over a built index.
//! 4. `edit_mixed` — interleaved insert/remove/update/query, the
//!    dynamics-shaped workload that motivated QBVH-native incremental.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use nalgebra::Point3;
use parsimony_spatial::{
    Aabb, BruteIndex, BvhIndex, QbvhIndex, Sphere, SpatialIndex, SpatialIndexExt,
};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000];
const LARGE_SIZES: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000];
const WORLD: f32 = 1_000.0;
const MAX_SIZE: f32 = 10.0;

fn gen_prims(n: usize, seed: u64) -> Vec<(u64, Aabb)> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let cx: f32 = rng.gen_range(-WORLD..WORLD);
            let cy: f32 = rng.gen_range(-WORLD..WORLD);
            let cz: f32 = rng.gen_range(-WORLD..WORLD);
            let hs: f32 = rng.gen_range(0.1..MAX_SIZE);
            (
                i as u64,
                Aabb::new(
                    Point3::new(cx - hs, cy - hs, cz - hs),
                    Point3::new(cx + hs, cy + hs, cz + hs),
                ),
            )
        })
        .collect()
}

fn gen_aabb_queries(n: usize, seed: u64, query_size: f32) -> Vec<Aabb> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let cx: f32 = rng.gen_range(-WORLD..WORLD);
            let cy: f32 = rng.gen_range(-WORLD..WORLD);
            let cz: f32 = rng.gen_range(-WORLD..WORLD);
            let hs = query_size;
            Aabb::new(
                Point3::new(cx - hs, cy - hs, cz - hs),
                Point3::new(cx + hs, cy + hs, cz + hs),
            )
        })
        .collect()
}

fn gen_sphere_queries(n: usize, seed: u64, radius: f32) -> Vec<Sphere> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            Sphere::new(
                Point3::new(
                    rng.gen_range(-WORLD..WORLD),
                    rng.gen_range(-WORLD..WORLD),
                    rng.gen_range(-WORLD..WORLD),
                ),
                radius,
            )
        })
        .collect()
}

// ---------- build ----------

fn bench_bulk_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_build");
    for &n in LARGE_SIZES {
        let prims = gen_prims(n, 0xC0DE);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("bvh", n), &prims, |b, prims| {
            b.iter(|| {
                let mut idx = BvhIndex::new();
                idx.build_from(prims.iter().copied()).unwrap();
                black_box(&idx);
            });
        });
        group.bench_with_input(BenchmarkId::new("qbvh", n), &prims, |b, prims| {
            b.iter(|| {
                let mut idx = QbvhIndex::new();
                idx.build_from(prims.iter().copied()).unwrap();
                black_box(&idx);
            });
        });
    }
    group.finish();
}

// ---------- incremental insert ----------

fn bench_incremental_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_insert");
    for &n in SIZES {
        let prims = gen_prims(n, 0xC0DE);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("bvh", n), &prims, |b, prims| {
            b.iter(|| {
                let mut idx = BvhIndex::new();
                for (uid, a) in prims {
                    idx.insert(*uid, *a).unwrap();
                }
                black_box(&idx);
            });
        });
        group.bench_with_input(BenchmarkId::new("qbvh", n), &prims, |b, prims| {
            b.iter(|| {
                let mut idx = QbvhIndex::new();
                for (uid, a) in prims {
                    idx.insert(*uid, *a).unwrap();
                }
                black_box(&idx);
            });
        });
    }
    group.finish();
}

// ---------- queries ----------

fn bench_query_aabb(c: &mut Criterion) {
    let queries = gen_aabb_queries(64, 0xBEEF, 20.0);
    let mut group = c.benchmark_group("query_aabb");
    for &n in SIZES {
        let prims = gen_prims(n, 0xC0DE);
        let mut bvh = BvhIndex::new();
        bvh.build_from(prims.iter().copied()).unwrap();
        let mut qbvh = QbvhIndex::new();
        qbvh.build_from(prims.iter().copied()).unwrap();
        let mut brute = BruteIndex::with_capacity(n);
        for (uid, a) in &prims {
            brute.insert(*uid, *a).unwrap();
        }

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(BenchmarkId::new("brute", n), &n, |b, _| {
            b.iter(|| {
                let mut total = 0usize;
                for q in &queries {
                    brute.query_aabb(q, |_| total += 1);
                }
                black_box(total);
            });
        });
        group.bench_with_input(BenchmarkId::new("bvh", n), &n, |b, _| {
            b.iter(|| {
                let mut total = 0usize;
                for q in &queries {
                    bvh.query_aabb(q, |_| total += 1);
                }
                black_box(total);
            });
        });
        group.bench_with_input(BenchmarkId::new("qbvh", n), &n, |b, _| {
            b.iter(|| {
                let mut total = 0usize;
                for q in &queries {
                    qbvh.query_aabb(q, |_| total += 1);
                }
                black_box(total);
            });
        });
    }
    group.finish();
}

fn bench_query_sphere(c: &mut Criterion) {
    let queries = gen_sphere_queries(64, 0xFEED, 20.0);
    let mut group = c.benchmark_group("query_sphere");
    for &n in SIZES {
        let prims = gen_prims(n, 0xC0DE);
        let mut bvh = BvhIndex::new();
        bvh.build_from(prims.iter().copied()).unwrap();
        let mut qbvh = QbvhIndex::new();
        qbvh.build_from(prims.iter().copied()).unwrap();
        let mut brute = BruteIndex::with_capacity(n);
        for (uid, a) in &prims {
            brute.insert(*uid, *a).unwrap();
        }

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(BenchmarkId::new("brute", n), &n, |b, _| {
            b.iter(|| {
                let mut total = 0usize;
                for q in &queries {
                    brute.query_sphere(q, |_| total += 1);
                }
                black_box(total);
            });
        });
        group.bench_with_input(BenchmarkId::new("bvh", n), &n, |b, _| {
            b.iter(|| {
                let mut total = 0usize;
                for q in &queries {
                    bvh.query_sphere(q, |_| total += 1);
                }
                black_box(total);
            });
        });
        group.bench_with_input(BenchmarkId::new("qbvh", n), &n, |b, _| {
            b.iter(|| {
                let mut total = 0usize;
                for q in &queries {
                    qbvh.query_sphere(q, |_| total += 1);
                }
                black_box(total);
            });
        });
    }
    group.finish();
}

// ---------- mixed edit + query (dynamics shape) ----------

/// One "step" of a dynamics-shaped workload: 10 mixed insert/remove/update
/// ops followed by 1 AABB query, repeated `STEPS_PER_ITER` times. This is
/// the workload that drove the QBVH-native incremental design.
const STEPS_PER_ITER: usize = 100;

fn build_index_for_edit_workload<I: SpatialIndex>(idx: &mut I, n: usize, seed: u64) -> Vec<u64> {
    let prims = gen_prims(n, seed);
    let mut live: Vec<u64> = Vec::with_capacity(n);
    for (uid, a) in &prims {
        idx.insert(*uid, *a).unwrap();
        live.push(*uid);
    }
    live
}

fn bench_edit_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("edit_mixed");
    // smaller sizes — this bench is heavier per iteration
    for &n in &[1_000usize, 10_000, 100_000] {
        let queries = gen_aabb_queries(STEPS_PER_ITER, 0xACE5, 20.0);
        let edits_seed = 0xED175u64;

        group.throughput(Throughput::Elements(STEPS_PER_ITER as u64));

        // BVH (binary): rebuild_if_needed after edits to keep query speed sane
        group.bench_with_input(BenchmarkId::new("bvh", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let mut idx = BvhIndex::new();
                    let live = build_index_for_edit_workload(&mut idx, n, 0xC0DE);
                    (idx, live)
                },
                |(mut idx, mut live)| {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(edits_seed);
                    let mut next_uid = n as u64;
                    for q in &queries {
                        // 10 mixed edits
                        for _ in 0..10 {
                            let r: f32 = rng.gen_range(0.0..1.0);
                            if r < 0.4 {
                                let uid = next_uid;
                                next_uid += 1;
                                let a = rand_query(&mut rng);
                                idx.insert(uid, a).unwrap();
                                live.push(uid);
                            } else if r < 0.7 && !live.is_empty() {
                                let i = rng.gen_range(0..live.len());
                                let uid = live.swap_remove(i);
                                idx.remove(uid).unwrap();
                            } else if !live.is_empty() {
                                let i = rng.gen_range(0..live.len());
                                let uid = live[i];
                                let a = rand_query(&mut rng);
                                idx.update(uid, a).unwrap();
                            }
                        }
                        // 1 query
                        let mut total = 0usize;
                        idx.query_aabb(q, |_| total += 1);
                        black_box(total);
                    }
                    black_box(&idx);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("qbvh", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let mut idx = QbvhIndex::new();
                    let live = build_index_for_edit_workload(&mut idx, n, 0xC0DE);
                    (idx, live)
                },
                |(mut idx, mut live)| {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(edits_seed);
                    let mut next_uid = n as u64;
                    for q in &queries {
                        for _ in 0..10 {
                            let r: f32 = rng.gen_range(0.0..1.0);
                            if r < 0.4 {
                                let uid = next_uid;
                                next_uid += 1;
                                let a = rand_query(&mut rng);
                                idx.insert(uid, a).unwrap();
                                live.push(uid);
                            } else if r < 0.7 && !live.is_empty() {
                                let i = rng.gen_range(0..live.len());
                                let uid = live.swap_remove(i);
                                idx.remove(uid).unwrap();
                            } else if !live.is_empty() {
                                let i = rng.gen_range(0..live.len());
                                let uid = live[i];
                                let a = rand_query(&mut rng);
                                idx.update(uid, a).unwrap();
                            }
                        }
                        let mut total = 0usize;
                        idx.query_aabb(q, |_| total += 1);
                        black_box(total);
                    }
                    black_box(&idx);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn rand_query<R: Rng>(rng: &mut R) -> Aabb {
    let cx: f32 = rng.gen_range(-WORLD..WORLD);
    let cy: f32 = rng.gen_range(-WORLD..WORLD);
    let cz: f32 = rng.gen_range(-WORLD..WORLD);
    let hs: f32 = rng.gen_range(0.1..MAX_SIZE);
    Aabb::new(
        Point3::new(cx - hs, cy - hs, cz - hs),
        Point3::new(cx + hs, cy + hs, cz + hs),
    )
}

// ---------- sanity check ----------

fn bench_sanity_check(c: &mut Criterion) {
    let prims = gen_prims(2_000, 0xABCD);
    let queries = gen_aabb_queries(20, 0xDCBA, 20.0);

    let mut bvh = BvhIndex::new();
    bvh.build_from(prims.iter().copied()).unwrap();
    let mut qbvh = QbvhIndex::new();
    qbvh.build_from(prims.iter().copied()).unwrap();
    let mut brute = BruteIndex::new();
    for (uid, a) in &prims {
        brute.insert(*uid, *a).unwrap();
    }
    for q in &queries {
        let mut g_bvh = bvh.collect_aabb(q);
        let mut g_qbvh = qbvh.collect_aabb(q);
        let mut g_brute = brute.collect_aabb(q);
        g_bvh.sort();
        g_qbvh.sort();
        g_brute.sort();
        assert_eq!(g_bvh, g_brute, "BVH and brute disagree pre-benchmark");
        assert_eq!(g_qbvh, g_brute, "QBVH and brute disagree pre-benchmark");
    }
    c.bench_function("sanity_2000", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                qbvh.query_aabb(q, |_| total += 1);
            }
            black_box(total);
        });
    });
}

criterion_group!(
    benches,
    bench_bulk_build,
    bench_incremental_insert,
    bench_query_aabb,
    bench_query_sphere,
    bench_edit_mixed,
    bench_sanity_check,
);
criterion_main!(benches);
