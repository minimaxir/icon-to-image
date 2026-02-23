//! Benchmarks for icon rendering at various sizes.
//!
//! Tests icon generation performance across a range of sizes from 16x16 to 2048x2048.
//! Includes 511x511 and 512x512 sizes specifically to measure SIMD performance
//! differences (512 is a multiple of 8, enabling potential AVX optimizations).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use icon_to_image::{Color, IconRenderer, RenderConfig};

/// Benchmark icon rendering at various sizes.
fn bench_render_sizes(c: &mut Criterion) {
    let renderer = IconRenderer::new().expect("Failed to create renderer");

    // Test sizes including SIMD boundary test cases (511 vs 512)
    let sizes: &[(u32, &str)] = &[
        (16, "16x16"),
        (32, "32x32"),
        (64, "64x64"),
        (128, "128x128"),
        (256, "256x256"),
        (511, "511x511 (non-SIMD)"),
        (512, "512x512 (SIMD-aligned)"),
        (1024, "1024x1024"),
        (2048, "2048x2048"),
    ];

    let mut group = c.benchmark_group("icon_render");

    for (size, label) in sizes {
        let pixel_count = (*size as u64) * (*size as u64);
        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(BenchmarkId::new("heart", label), size, |b, &size| {
            let config = RenderConfig::new()
                .canvas_size(size, size)
                .icon_size(((size as f64) * 0.95) as u32)
                .supersample(1)
                .icon_color(Color::rgb(255, 0, 0))
                .background_color(Color::white());

            b.iter(|| {
                let result = renderer.render(black_box("heart"), black_box(&config));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark with supersampling enabled (2x).
fn bench_render_supersampled(c: &mut Criterion) {
    let renderer = IconRenderer::new().expect("Failed to create renderer");

    let sizes: &[(u32, &str)] = &[
        (256, "256x256 (2x SS)"),
        (511, "511x511 (2x SS, non-SIMD)"),
        (512, "512x512 (2x SS, SIMD-aligned)"),
    ];

    let mut group = c.benchmark_group("icon_render_supersampled");

    for (size, label) in sizes {
        let pixel_count = (*size as u64) * (*size as u64);
        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(BenchmarkId::new("heart", label), size, |b, &size| {
            let config = RenderConfig::new()
                .canvas_size(size, size)
                .icon_size(((size as f64) * 0.95) as u32)
                .supersample(2)
                .icon_color(Color::rgb(255, 0, 0))
                .background_color(Color::white());

            b.iter(|| {
                let result = renderer.render(black_box("heart"), black_box(&config));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark with rotation enabled.
fn bench_render_rotated(c: &mut Criterion) {
    let renderer = IconRenderer::new().expect("Failed to create renderer");

    let sizes: &[(u32, &str)] = &[(256, "256x256 (rotated)"), (512, "512x512 (rotated)")];

    let mut group = c.benchmark_group("icon_render_rotated");

    for (size, label) in sizes {
        let pixel_count = (*size as u64) * (*size as u64);
        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(BenchmarkId::new("heart_45deg", label), size, |b, &size| {
            let config = RenderConfig::new()
                .canvas_size(size, size)
                .icon_size(((size as f64) * 0.95) as u32)
                .supersample(1)
                .rotate(45.0)
                .icon_color(Color::rgb(255, 0, 0))
                .background_color(Color::white());

            b.iter(|| {
                let result = renderer.render(black_box("heart"), black_box(&config));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark comparing different icons (complexity varies).
fn bench_render_different_icons(c: &mut Criterion) {
    let renderer = IconRenderer::new().expect("Failed to create renderer");
    let size = 512u32;

    let icons = &["heart", "star", "circle", "square", "check"];

    let mut group = c.benchmark_group("icon_complexity");
    group.throughput(Throughput::Elements((size as u64) * (size as u64)));

    for icon in icons {
        group.bench_with_input(BenchmarkId::new("512x512", icon), icon, |b, &icon| {
            let config = RenderConfig::new()
                .canvas_size(size, size)
                .icon_size(486)
                .supersample(1)
                .icon_color(Color::black())
                .background_color(Color::white());

            b.iter(|| {
                let result = renderer.render(black_box(icon), black_box(&config));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_render_sizes,
    bench_render_supersampled,
    bench_render_rotated,
    bench_render_different_icons
);
criterion_main!(benches);
