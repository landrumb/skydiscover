#[cfg(target_arch = "x86_64")]
use scratch::data_handling::dataset::VectorDataset;
#[cfg(target_arch = "x86_64")]
use scratch::data_handling::rabitq_fast_scan::{RabitqFastScan, BLOCK_SIZE};
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
const DIM: usize = 1536;
#[cfg(target_arch = "x86_64")]
const N_VECS: usize = 1024 * 64;
#[cfg(target_arch = "x86_64")]
const N_QUERIES: usize = 64;
#[cfg(target_arch = "x86_64")]
const OUTER_ITERS: usize = 6;
#[cfg(target_arch = "x86_64")]
const WARMUP_QUERIES: usize = 2;

#[inline]
#[cfg(target_arch = "x86_64")]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
#[cfg(target_arch = "x86_64")]
fn random_f32(seed: u64) -> f32 {
    let bits = splitmix64(seed);
    let unit = (bits as f64) / (u64::MAX as f64);
    (unit * 2.0 - 1.0) as f32
}

#[cfg(target_arch = "x86_64")]
fn make_random_dataset(n: usize, dim: usize, seed: u64) -> VectorDataset<f32> {
    let mut data = vec![0.0f32; n * dim];
    for i in 0..n {
        let base = seed ^ ((i as u64) << 32);
        for d in 0..dim {
            data[i * dim + d] = random_f32(base ^ d as u64);
        }
    }
    VectorDataset::new(data.into_boxed_slice(), n, dim)
}

#[cfg(target_arch = "x86_64")]
fn make_random_query(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|d| random_f32(seed ^ ((d as u64) << 1)))
        .collect()
}

#[inline]
#[cfg(target_arch = "x86_64")]
fn update_checksum(mut checksum: u64, value: u64) -> u64 {
    checksum ^= value.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    checksum = checksum.rotate_left(27);
    checksum.wrapping_mul(0x94D0_49BB_1331_11EB)
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!(
            "bench_rabitq_fast_scan_distance requires AVX2. \
             Rebuild with RUSTFLAGS='-C target-cpu=native' on an AVX2-capable machine."
        );
        std::process::exit(1);
    }

    let dataset = make_random_dataset(N_VECS, DIM, 0x1234_5678);
    let fastscan = RabitqFastScan::<DIM>::from_f32_dataset(&dataset);
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|i| make_random_query(DIM, 0xABC0_0000 + i as u64))
        .collect();

    for query in queries.iter().take(WARMUP_QUERIES.min(queries.len())) {
        let oracle = fastscan.make_oracle(query);
        let mut checksum = 0u64;
        for block_idx in 0..fastscan.num_blocks() {
            let block = oracle.compute_block_distances(block_idx);
            checksum = update_checksum(checksum, block_idx as u64);
            for (idx, dist) in block {
                checksum = update_checksum(checksum, ((idx as u64) << 32) ^ dist.to_bits() as u64);
            }
        }
        std::hint::black_box(checksum);
    }

    let total_queries = N_QUERIES * OUTER_ITERS;
    let start = Instant::now();
    let mut checksum = 0u64;

    for iter_idx in 0..OUTER_ITERS {
        for (query_idx, query) in queries.iter().enumerate() {
            checksum = update_checksum(checksum, ((iter_idx as u64) << 32) ^ query_idx as u64);
            let oracle = fastscan.make_oracle(query);
            for block_idx in 0..fastscan.num_blocks() {
                let block = oracle.compute_block_distances(block_idx);
                checksum = update_checksum(checksum, block_idx as u64);
                for (idx, dist) in block {
                    checksum =
                        update_checksum(checksum, ((idx as u64) << 32) ^ dist.to_bits() as u64);
                }
            }
        }
    }

    let elapsed = start.elapsed();
    let qps = total_queries as f64 / elapsed.as_secs_f64();
    let avg_query_ms = elapsed.as_secs_f64() * 1000.0 / total_queries as f64;

    println!("QPS={qps:.6}");
    println!("AVG_QUERY_MS={avg_query_ms:.6}");
    println!("AVG_BLOCKS_PER_QUERY={}", fastscan.num_blocks());
    println!("CHECKSUM={checksum:016x}");
    println!("DIM={DIM}");
    println!("N_VECS={N_VECS}");
    println!("BLOCK_SIZE={BLOCK_SIZE}");
    println!("N_QUERIES={N_QUERIES}");
    println!("OUTER_ITERS={OUTER_ITERS}");
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("bench_rabitq_fast_scan_distance targets the x86_64 AVX2 path.");
    std::process::exit(1);
}
