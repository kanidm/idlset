extern crate criterion;
extern crate idlset;

mod idl_simple;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use idl_simple::IDLSimple;
use idlset::IDLBitRange;
use idlset::v2::IDLBitRange as IDLBitRangeV2;
use std::iter::FromIterator;

// Trying to make these work with trait bounds is literally too hard
// So just make our own impls.

struct Duplex(Vec<u64>, Vec<u64>);

struct SDuplex(IDLSimple, IDLSimple);

impl std::fmt::Display for SDuplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} - {}", self.0.len(), self.1.len())
    }
}

struct RDuplex(IDLBitRange, IDLBitRange);

impl std::fmt::Display for RDuplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} - {}", self.0.len(), self.1.len())
    }
}

struct V2Duplex(IDLBitRangeV2, IDLBitRangeV2);

impl std::fmt::Display for V2Duplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} - {} {}", self.0.len(), self.0.is_compressed(), self.1.len(), self.1.is_compressed())
    }
}


struct Triplex(Vec<u64>, Vec<u64>, Vec<u64>);

struct STriplex(IDLSimple, IDLSimple, IDLSimple);

impl std::fmt::Display for STriplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -- {} -- {}",
            self.0.len(),
            self.1.len(),
            self.2.len()
        )
    }
}

struct RTriplex(IDLBitRange, IDLBitRange, IDLBitRange);

impl std::fmt::Display for RTriplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -- {} -- {}",
            self.0.len(),
            self.1.len(),
            self.2.len()
        )
    }
}

struct SComplex(IDLSimple, IDLSimple, Vec<IDLSimple>);

impl std::fmt::Display for SComplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -- {} -- [{}]",
            self.0.len(),
            self.1.len(),
            self.2.len()
        )
    }
}

struct RComplex(IDLBitRange, IDLBitRange, Vec<IDLBitRange>);

impl std::fmt::Display for RComplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -- {} -- [{}]",
            self.0.len(),
            self.1.len(),
            self.2.len()
        )
    }
}

// =========

fn do_bench_duplex(c: &mut Criterion, label: &str, i: Duplex) {
    let mut group = c.benchmark_group(&format!("{}_union", label));

    let si = SDuplex(
        IDLSimple::from_iter(i.0.clone()),
        IDLSimple::from_iter(i.1.clone()),
    );
    let ri = RDuplex(
        IDLBitRange::from_iter(i.0.clone()),
        IDLBitRange::from_iter(i.1.clone()),
    );

    let mut v2i = V2Duplex(
        IDLBitRangeV2::from_iter(i.0.clone()),
        IDLBitRangeV2::from_iter(i.1.clone()),
    );

    group.bench_with_input(BenchmarkId::new("Simple", &si), &si, |t, SDuplex(a, b)| {
        t.iter(|| { b | a }.sum())
    });
    group.bench_with_input(
        BenchmarkId::new("Compressed", &ri),
        &ri,
        |t, RDuplex(a, b)| t.iter(|| { b | a }.sum()),
    );
    group.bench_with_input(
        BenchmarkId::new("Compressed V2 Sparse Sparse", &ri),
        &ri,
        |t, V2Duplex(a, b)| t.iter(|| { b | a }.sum()),
    );

    v2i.0.compress();
    group.bench_with_input(
        BenchmarkId::new("Compressed V2 Sparse Compressed", &ri),
        &ri,
        |t, V2Duplex(a, b)| t.iter(|| { b | a }.sum()),
    );

    v2i.1.compress();
    group.bench_with_input(
        BenchmarkId::new("Compressed V2 Compressed Compressed", &ri),
        &ri,
        |t, V2Duplex(a, b)| t.iter(|| { b | a }.sum()),
    );
    group.finish();

    let mut v2i = V2Duplex(
        IDLBitRangeV2::from_iter(i.0.clone()),
        IDLBitRangeV2::from_iter(i.1.clone()),
    );

    let mut group = c.benchmark_group(&format!("{}_intersection", label));
    group.bench_with_input(BenchmarkId::new("Simple", &si), &si, |t, SDuplex(a, b)| {
        t.iter(|| { b & a }.sum())
    });
    group.bench_with_input(
        BenchmarkId::new("Compressed", &ri),
        &ri,
        |t, RDuplex(a, b)| t.iter(|| { b & a }.sum()),
    );

    group.bench_with_input(
        BenchmarkId::new("Compressed V2 Sparse Sparse", &ri),
        &ri,
        |t, V2Duplex(a, b)| t.iter(|| { b & a }.sum()),
    );

    v2i.0.compress();
    group.bench_with_input(
        BenchmarkId::new("Compressed V2 Sparse Compressed", &ri),
        &ri,
        |t, V2Duplex(a, b)| t.iter(|| { b & a }.sum()),
    );

    v2i.1.compress();
    group.bench_with_input(
        BenchmarkId::new("Compressed V2 Compressed Compressed", &ri),
        &ri,
        |t, V2Duplex(a, b)| t.iter(|| { b & a }.sum()),
    );

    group.finish();
}

fn bench_duplex(c: &mut Criterion) {
    let i = Duplex(
        vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900],
        Vec::from_iter(1..1024),
    );
    do_bench_duplex(c, "1_small", i);

    let i = Duplex(vec![1], Vec::from_iter(1..102400));
    do_bench_duplex(c, "2_large_early", i);

    let i = Duplex(vec![102399], Vec::from_iter(1..102400));
    do_bench_duplex(c, "3_large_deep", i);

    let i = Duplex(Vec::from_iter(1..1024), Vec::from_iter(1..1024));
    do_bench_duplex(c, "4_mod_eq", i);

    let i = Duplex(Vec::from_iter(1..102400), Vec::from_iter(1..102400));
    do_bench_duplex(c, "5_large_eq", i);

    let i = Duplex(vec![1], vec![1]);
    do_bench_duplex(c, "6_small_eq", i);

    let i = Duplex(vec![1], vec![2]);
    do_bench_duplex(c, "7_small_neq", i);

    let i = Duplex(vec![16], Vec::from_iter(1..32));
    do_bench_duplex(c, "8_ludwig_small", i);

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64 * i + 5)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64 * i + 5)
    }
    let i = Duplex(vec1, vec2);
    do_bench_duplex(c, "9_ludwig_sparse", i);

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64 * i + 5);
        vec1.push(64 * i + 15)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64 * i + 5);
        vec2.push(64 * i + 15)
    }

    let i = Duplex(vec1, vec2);
    do_bench_duplex(c, "10_ludwig_sparse", i);

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64 * i + 5);
        vec1.push(64 * i + 7);
        vec1.push(64 * i + 15)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64 * i + 5);
        vec2.push(64 * i + 7);
        vec2.push(64 * i + 15)
    }

    let i = Duplex(vec1, vec2);
    do_bench_duplex(c, "11_ludwig_sparse", i);

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64 * i + 5);
        vec1.push(64 * i + 7);
        vec1.push(64 * i + 15);
        vec1.push(64 * i + 20);
        vec1.push(64 * i + 25);
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64 * i + 5);
        vec2.push(64 * i + 7);
        vec2.push(64 * i + 15);
        vec2.push(64 * i + 20);
        vec2.push(64 * i + 25);
    }

    let i = Duplex(vec1, vec2);
    do_bench_duplex(c, "12_ludwig_sparse", i);

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64 * i + 5);
        vec1.push(64 * i + 7);
        vec1.push(64 * i + 15);
        vec1.push(64 * i + 20);
        vec1.push(64 * i + 25);
        vec1.push(64 * i + 30);
        vec1.push(64 * i + 35);
        vec1.push(64 * i + 40);
        vec1.push(64 * i + 45);
        vec1.push(64 * i + 50);
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64 * i + 5);
        vec2.push(64 * i + 7);
        vec2.push(64 * i + 15);
        vec2.push(64 * i + 20);
        vec2.push(64 * i + 25);
        vec1.push(64 * i + 30);
        vec1.push(64 * i + 35);
        vec1.push(64 * i + 40);
        vec1.push(64 * i + 45);
        vec1.push(64 * i + 50);
    }

    let i = Duplex(vec1, vec2);
    do_bench_duplex(c, "13_ludwig_sparse", i);
}

fn do_bench_triplex(c: &mut Criterion, label: &str, i: Triplex) {
    let si = STriplex(
        IDLSimple::from_iter(i.0.clone()),
        IDLSimple::from_iter(i.1.clone()),
        IDLSimple::from_iter(i.2.clone()),
    );
    let ri = RTriplex(
        IDLBitRange::from_iter(i.0.clone()),
        IDLBitRange::from_iter(i.1.clone()),
        IDLBitRange::from_iter(i.2.clone()),
    );

    let mut group = c.benchmark_group(&format!("{}_union", label));
    group.bench_with_input(
        BenchmarkId::new("Simple", &si),
        &si,
        |t, STriplex(a, b, c)| t.iter(|| { &(a | b) | c }.sum()),
    );
    group.bench_with_input(
        BenchmarkId::new("Compressed", &ri),
        &ri,
        |t, RTriplex(a, b, c)| t.iter(|| { &(a | b) | c }.sum()),
    );
    group.finish();

    let mut group = c.benchmark_group(&format!("{}_intersection", label));
    group.bench_with_input(
        BenchmarkId::new("Simple", &si),
        &si,
        |t, STriplex(a, b, c)| t.iter(|| { &(a & b) & c }.sum()),
    );
    group.bench_with_input(
        BenchmarkId::new("Compressed", &ri),
        &ri,
        |t, RTriplex(a, b, c)| t.iter(|| { &(a & b) & c }.sum()),
    );
    group.finish();

    let mut group = c.benchmark_group(&format!("{}_intersection_union", label));
    group.bench_with_input(
        BenchmarkId::new("Simple", &si),
        &si,
        |t, STriplex(a, b, c)| t.iter(|| { &(a & b) | c }.sum()),
    );
    group.bench_with_input(
        BenchmarkId::new("Compressed", &ri),
        &ri,
        |t, RTriplex(a, b, c)| t.iter(|| { &(a & b) | c }.sum()),
    );
    group.finish();

    let mut group = c.benchmark_group(&format!("{}_union_intersection", label));
    group.bench_with_input(
        BenchmarkId::new("Simple", &si),
        &si,
        |t, STriplex(a, b, c)| t.iter(|| { &(a | b) & c }.sum()),
    );
    group.bench_with_input(
        BenchmarkId::new("Compressed", &ri),
        &ri,
        |t, RTriplex(a, b, c)| t.iter(|| { &(a | b) & c }.sum()),
    );
    group.finish();
}

fn bench_triplex(c: &mut Criterion) {
    do_bench_triplex(
        c,
        "1_trip_small",
        Triplex(
            vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900],
            Vec::from_iter(1..1024),
            vec![2, 3, 35, 64, 128, 150, 152, 180, 256, 900, 1024, 1500, 1600],
        ),
    );

    do_bench_triplex(
        c,
        "2_trip_large",
        Triplex(
            Vec::from_iter(1..102400),
            Vec::from_iter(1..102400),
            Vec::from_iter(1..102400),
        ),
    );

    do_bench_triplex(
        c,
        "3_trip_large_overlap",
        Triplex(
            Vec::from_iter(1..102400),
            Vec::from_iter(81920..184320),
            Vec::from_iter(160240..242160),
        ),
    );

    do_bench_triplex(
        c,
        "4_trip_large_sparse",
        Triplex(
            Vec::from_iter(1..102400),
            Vec::from_iter(1..102400),
            vec![2, 3, 35, 64, 128, 150, 152, 180, 256, 900, 1024, 1500, 1600],
        ),
    );

    do_bench_triplex(
        c,
        "5_trip_large_sparse",
        Triplex(
            Vec::from_iter(1..102400),
            Vec::from_iter(1..102400),
            vec![40960],
        ),
    );

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64 * i + 5);
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64 * i + 5);
    }
    let mut vec3 = Vec::new();
    for i in 400..700 {
        vec3.push(64 * i + 5);
    }
    do_bench_triplex(c, "6_trip_sparse_overlap", Triplex(vec1, vec2, vec3));

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64 * i + 5);
        vec1.push(64 * i + 15)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64 * i + 5);
        vec2.push(64 * i + 15)
    }
    let mut vec3 = Vec::new();
    for i in 400..700 {
        vec3.push(64 * i + 5);
        vec3.push(64 * i + 15)
    }
    do_bench_triplex(c, "7_trip_sparse_overlap", Triplex(vec1, vec2, vec3));

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64 * i + 5);
        vec1.push(64 * i + 7);
        vec1.push(64 * i + 15)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64 * i + 5);
        vec2.push(64 * i + 7);
        vec2.push(64 * i + 15)
    }
    let mut vec3 = Vec::new();
    for i in 400..700 {
        vec3.push(64 * i + 5);
        vec3.push(64 * i + 7);
        vec3.push(64 * i + 15)
    }
    do_bench_triplex(c, "8_trip_sparse_overlap", Triplex(vec1, vec2, vec3));
}

fn do_bench_complex(c: &mut Criterion, label: &str, i: Triplex) {
    let mut group = c.benchmark_group(&format!("{}_A", label));

    let si = SComplex(
        IDLSimple::from_iter(i.0.clone()),
        IDLSimple::from_iter(i.1.clone()),
        i.2.iter().map(|&x| IDLSimple::from_u64(x)).collect(),
    );
    let ri = RComplex(
        IDLBitRange::from_iter(i.0.clone()),
        IDLBitRange::from_iter(i.1.clone()),
        i.2.iter().map(|&x| IDLBitRange::from_u64(x)).collect(),
    );

    group.bench_with_input(
        BenchmarkId::new("Simple", &si),
        &si,
        |t, SComplex(a, b, c)| {
            t.iter(|| {
                let idl_inter = c.iter().fold(IDLSimple::new(), |acc, x| &acc | x);
                { &(a & b) & &idl_inter }.sum()
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("Compressed", &ri),
        &ri,
        |t, RComplex(a, b, c)| {
            t.iter(|| {
                let idl_inter = c.iter().fold(IDLBitRange::new(), |acc, x| &acc | x);
                { &(a & b) & &idl_inter }.sum()
            })
        },
    );

    group.finish();
}

fn bench_complex(c: &mut Criterion) {
    do_bench_complex(
        c,
        "1_complex_realistic",
        Triplex(
            Vec::from_iter(1..102400),
            Vec::from_iter(51200..102400),
            vec![
                2, 3, 35, 64, 128, 150, 152, 180, 256, 900, 1024, 1500, 1600, 2400, 2401, 2403,
                4500, 7890, 10000, 40000, 78900,
            ],
        ),
    );
}

criterion_group!(idlbenches, bench_duplex, bench_triplex, bench_complex);
criterion_main!(idlbenches);
