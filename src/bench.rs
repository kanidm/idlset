
extern crate idlset;
extern crate time;

use idlset::idl_simple::IDLSimple;
use idlset::idl_range::IDLBitRange;
use std::iter::FromIterator;

// Trying to make these work with trait bounds is literally too hard
// So just make our own impls.

fn simple_consume_results(idl: &IDLSimple) -> u64
{
    let mut result: u64 = 0;
    for id in idl {
        result += id;
    }
    return result;
}

fn range_consume_results(idl: &IDLBitRange) -> u64
{
    let mut result: u64 = 0;
    for id in idl {
        result += id;
    }
    return result;
}

fn bench_simple_union(id: &str, a: Vec<u64>, b: Vec<u64>) {
    let idl_a = IDLSimple::from_iter(a);
    let idl_b = IDLSimple::from_iter(b);

    let start = time::now();
    let idl_result = idl_a | idl_b;
    let result = simple_consume_results(&idl_result);
    let end = time::now();
    println!("simple union {}: {} -> {}", id, end - start, result);
}

fn bench_range_union(id: &str, a: Vec<u64>, b: Vec<u64>) {
    let idl_a = IDLBitRange::from_iter(a);
    let idl_b = IDLBitRange::from_iter(b);

    let start = time::now();
    let idl_result = idl_a | idl_b;
    let result = range_consume_results(&idl_result);
    let end = time::now();
    println!("range union  {}: {} -> {}", id, end - start, result);
}


fn bench_simple_intersection(id: &str, a: Vec<u64>, b: Vec<u64>) {
    let idl_a = IDLSimple::from_iter(a);
    let idl_b = IDLSimple::from_iter(b);

    let start = time::now();
    let idl_result = idl_a & idl_b;
    let result = simple_consume_results(&idl_result);
    let end = time::now();
    println!("simple inter {}: {} -> {}", id, end - start, result);
}

fn bench_range_intersection(id: &str, a: Vec<u64>, b: Vec<u64>) {
    let idl_a = IDLBitRange::from_iter(a);
    let idl_b = IDLBitRange::from_iter(b);

    let start = time::now();
    let idl_result = idl_a & idl_b;
    let result = range_consume_results(&idl_result);
    let end = time::now();
    println!("range inter  {}: {} -> {}", id, end - start, result);
}

fn test_duplex(id: &str, a: Vec<u64>, b: Vec<u64>) {
    bench_simple_intersection(id, a.clone(), b.clone());
    bench_range_intersection(id, a.clone(), b.clone());
    bench_simple_union(id, a.clone(), b.clone());
    bench_range_union(id, a.clone(), b.clone());
    println!("=====");
}

fn bench_t_simple_union(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    let idl_a = IDLSimple::from_iter(a);
    let idl_b = IDLSimple::from_iter(b);
    let idl_c = IDLSimple::from_iter(c);

    let start = time::now();
    let idl_result = idl_a | idl_b | idl_c;
    let result = simple_consume_results(&idl_result);
    let end = time::now();
    println!("simple 3 union {}: {} -> {}", id, end - start, result);
}

fn bench_t_range_union(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    let idl_a = IDLBitRange::from_iter(a);
    let idl_b = IDLBitRange::from_iter(b);
    let idl_c = IDLBitRange::from_iter(c);

    let start = time::now();
    let idl_result = idl_a | idl_b | idl_c;
    let result = range_consume_results(&idl_result);
    let end = time::now();
    println!("range 3  union {}: {} -> {}", id, end - start, result);
}

fn bench_t_simple_intersection(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    let idl_a = IDLSimple::from_iter(a);
    let idl_b = IDLSimple::from_iter(b);
    let idl_c = IDLSimple::from_iter(c);

    let start = time::now();
    let idl_result = idl_a & idl_b & idl_c;
    let result = simple_consume_results(&idl_result);
    let end = time::now();
    println!("simple 3 inter {}: {} -> {}", id, end - start, result);
}

fn bench_t_range_intersection(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    let idl_a = IDLBitRange::from_iter(a);
    let idl_b = IDLBitRange::from_iter(b);
    let idl_c = IDLBitRange::from_iter(c);

    let start = time::now();
    let idl_result = idl_a & idl_b & idl_c;
    let result = range_consume_results(&idl_result);
    let end = time::now();
    println!("range 3  inter {}: {} -> {}", id, end - start, result);
}

fn bench_t_simple_inter_union(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    let idl_a = IDLSimple::from_iter(a);
    let idl_b = IDLSimple::from_iter(b);
    let idl_c = IDLSimple::from_iter(c);

    let start = time::now();
    let idl_result = idl_a & idl_b | idl_c;
    let result = simple_consume_results(&idl_result);
    let end = time::now();
    println!("simple 3 in/un {}: {} -> {}", id, end - start, result);
}

fn bench_t_range_inter_union(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    let idl_a = IDLBitRange::from_iter(a);
    let idl_b = IDLBitRange::from_iter(b);
    let idl_c = IDLBitRange::from_iter(c);

    let start = time::now();
    let idl_result = idl_a & idl_b | idl_c;
    let result = range_consume_results(&idl_result);
    let end = time::now();
    println!("range 3  in/un {}: {} -> {}", id, end - start, result);
}

fn test_triplex(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    bench_t_simple_intersection(id, a.clone(), b.clone(), c.clone());
    bench_t_range_intersection(id, a.clone(), b.clone(), c.clone());
    bench_t_simple_union(id, a.clone(), b.clone(), c.clone());
    bench_t_range_union(id, a.clone(), b.clone(), c.clone());
    bench_t_simple_inter_union(id, a.clone(), b.clone(), c.clone());
    bench_t_range_inter_union(id, a.clone(), b.clone(), c.clone());
    bench_t_simple_inter_union(id, c.clone(), b.clone(), a.clone());
    bench_t_range_inter_union(id, c.clone(), b.clone(), a.clone());
    println!("=====");
}

fn bench_c_range(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    let idl_a = IDLBitRange::from_iter(a);
    let idl_b = IDLBitRange::from_iter(b);
    let uids: Vec<IDLBitRange> = c.iter()
                                  .map(|&x| IDLBitRange::from_u64(x) )
                                  .collect();

    let start = time::now();
    let mut uid_iter = uids.into_iter();

    // Get the first range
    let idl_start = uid_iter.next().unwrap();

    let idl_inter = uid_iter.fold(idl_start, |acc, x| acc | x);

    let idl_result = idl_a & idl_b & idl_inter;
    let result = range_consume_results(&idl_result);
    let end = time::now();
    println!("range 3  comp  {}: {} -> {}", id, end - start, result);
}

fn bench_c_simple(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    let idl_a = IDLSimple::from_iter(a);
    let idl_b = IDLSimple::from_iter(b);
    let uids: Vec<IDLSimple> = c.iter()
                                  .map(|&x| IDLSimple::from_u64(x) )
                                  .collect();

    let start = time::now();
    let mut uid_iter = uids.into_iter();

    // Get the first range
    let idl_start = uid_iter.next().unwrap();

    let idl_inter = uid_iter.fold(idl_start, |acc, x| acc | x);

    let idl_result = idl_a & idl_b & idl_inter;
    let result = simple_consume_results(&idl_result);
    let end = time::now();
    println!("simple 3 comp  {}: {} -> {}", id, end - start, result);
}

fn test_complex(id: &str, a: Vec<u64>, b: Vec<u64>, c: Vec<u64>) {
    bench_c_simple(id, a.clone(), b.clone(), c.clone());
    bench_c_range(id, a.clone(), b.clone(), c.clone());
}

fn main() {
    test_duplex(
        "1",
        vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900],
        Vec::from_iter(1..1024)
    );
    test_duplex(
        "2",
        vec![1],
        Vec::from_iter(1..102400)
    );
    test_duplex(
        "3",
        vec![102399],
        Vec::from_iter(1..102400)
    );
    test_duplex(
        "4",
        Vec::from_iter(1..1024),
        Vec::from_iter(1..1024)
    );
    test_duplex(
        "5",
        Vec::from_iter(1..102400),
        Vec::from_iter(1..102400)
    );
    test_duplex(
        "6",
        vec![1],
        vec![1],
    );
    test_duplex(
        "7",
        vec![1],
        vec![2],
    );
    /*
     * lkrispens tests. These show issues with sparse semi-overlapping
     * ranges. This mainly affects the low-population test 8.
     */
    test_duplex(
        "8",
        vec![16],
        Vec::from_iter(1..32)
    );
    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64*i + 5)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64*i + 5)
    }
    test_duplex(
        "9",
        vec1,
        vec2
    );
    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64*i + 5);
        vec1.push(64*i + 15)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64*i + 5);
        vec2.push(64*i + 15)
    }
    test_duplex(
        "10",
        vec1,
        vec2
    );
    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64*i + 5);
        vec1.push(64*i + 7);
        vec1.push(64*i + 15)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64*i + 5);
        vec2.push(64*i + 7);
        vec2.push(64*i + 15)
    }
    test_duplex(
        "11",
        vec1,
        vec2
    );

    test_triplex(
        "trip: 1",
        vec![2, 3, 8, 35, 64, 128, 130, 150, 152, 180, 256, 800, 900],
        Vec::from_iter(1..1024),
        vec![2, 3, 35, 64, 128, 150, 152, 180, 256, 900, 1024, 1500, 1600],
    );
    test_triplex(
        "trip: 2",
        Vec::from_iter(1..102400),
        Vec::from_iter(1..102400),
        Vec::from_iter(1..102400)
    );
    test_triplex(
        "trip: 3",
        Vec::from_iter(1..102400),
        Vec::from_iter(81920..184320),
        Vec::from_iter(160240..242160)
    );
    test_triplex(
        "trip: 4",
        Vec::from_iter(1..102400),
        Vec::from_iter(1..102400),
        vec![2, 3, 35, 64, 128, 150, 152, 180, 256, 900, 1024, 1500, 1600],
    );
    test_triplex(
        "trip: 5",
        Vec::from_iter(1..102400),
        Vec::from_iter(1..102400),
        vec![40960],
    );

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64*i + 5);
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64*i + 5);
    }
    let mut vec3 = Vec::new();
    for i in 400..700 {
        vec3.push(64*i + 5);
    }
    test_triplex(
        "trip: 6",
        vec1,
        vec2,
        vec3,
    );

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64*i + 5);
        vec1.push(64*i + 15)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64*i + 5);
        vec2.push(64*i + 15)
    }
    let mut vec3 = Vec::new();
    for i in 400..700 {
        vec3.push(64*i + 5);
        vec3.push(64*i + 15)
    }
    test_triplex(
        "trip: 7",
        vec1,
        vec2,
        vec3,
    );

    let mut vec1 = Vec::new();
    for i in 1..300 {
        vec1.push(64*i + 5);
        vec1.push(64*i + 7);
        vec1.push(64*i + 15)
    }
    let mut vec2 = Vec::new();
    for i in 200..500 {
        vec2.push(64*i + 5);
        vec2.push(64*i + 7);
        vec2.push(64*i + 15)
    }
    let mut vec3 = Vec::new();
    for i in 400..700 {
        vec3.push(64*i + 5);
        vec3.push(64*i + 7);
        vec3.push(64*i + 15)
    }
    test_triplex(
        "trip: 8",
        vec1,
        vec2,
        vec3,
    );

    test_complex(
        "comp: 1",
        Vec::from_iter(1..102400),
        Vec::from_iter(51200..102400),
        vec![2, 3, 35, 64, 128, 150, 152, 180, 256, 900, 1024, 1500, 1600, 2400, 2401, 2403, 4500, 7890, 10000, 40000, 78900],
    );
}


