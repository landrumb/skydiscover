// This file is spliced into source_template.rs by evaluator.py

// EVOLVE-BLOCK-START
    /// Computes raw FastScan results for a single block.
    ///
    /// Returns the raw u16 inner product accumulations for 32 vectors.
    #[inline]
    pub fn compute_block_raw(&self, block_idx: usize) -> [u16; BLOCK_SIZE] {
        debug_assert!(
            block_idx < self.fastscan.n_blocks,
            "Block index {} out of bounds (total blocks {})",
            block_idx,
            self.fastscan.n_blocks
        );

        let block_size_bytes = self.fastscan.block_size_bytes();
        let code_start = block_idx * block_size_bytes;

        let mut results = [0u16; BLOCK_SIZE];

        // Use pointer arithmetic to avoid bounds checking in the hot path
        unsafe {
            let codes_ptr = self.fastscan.packed_codes.as_ptr().add(code_start);
            let codes_slice = std::slice::from_raw_parts(codes_ptr, block_size_bytes);
            rabitq_fast_scan_kernels::fast_scan_dispatch::<B>(
                codes_slice,
                &self.luts,
                &mut results,
            );
        }

        results
    }

    /// Computes approximate L2 distances for a single block.
    ///
    /// Uses the RaBitQ distance estimation formula:
    /// `est_dist = f_add + g_add + f_rescale * (ip_result + g_k1xsumq)`
    ///
    /// Returns 32 (index, distance) pairs.
    // #[inline]  // Removed to allow instrumentation
    #[fastrace::trace(name = "compare")]
    pub fn compute_block_distances(&self, block_idx: usize) -> [(usize, f32); BLOCK_SIZE] {
        let raw_results = self.compute_block_raw(block_idx);
        let base_idx = block_idx * BLOCK_SIZE;

        let mut output = [(0usize, 0.0f32); BLOCK_SIZE];

        // Precompute query-side terms
        let g_add = self.g_add;
        let g_k1xsumq = self.g_k1xsumq;
        let lut_delta = self.lut_delta;
        let lut_sum_vl = self.lut_sum_vl;

        // Increment distance comparison counter if dcmp feature is enabled
        #[cfg(feature = "dcmp")]
        {
            let count = if base_idx + BLOCK_SIZE <= self.fastscan.n {
                BLOCK_SIZE
            } else if base_idx < self.fastscan.n {
                self.fastscan.n - base_idx
            } else {
                0
            };
            if count > 0 {
                crate::distance::increment_distance_comparison_count_by(count as u64);
            }
        }

        for i in 0..BLOCK_SIZE {
            let vec_idx = base_idx + i;
            let raw_ip = unsafe { *raw_results.get_unchecked(i) } as f32;

            // Dequantize the inner product result using precomputed parameters
            // ip_float = delta * raw_ip + sum_vl
            let ip_float = lut_delta * raw_ip + lut_sum_vl;

            // Get per-vector factors (use get_unchecked in release mode)
            let factors = unsafe { self.fastscan.factors.get_unchecked(vec_idx) };

            // RaBitQ distance estimate:
            // est_dist = f_add + g_add + f_rescale * (ip_float + g_k1xsumq)
            let est_dist = factors.f_add + g_add + factors.f_rescale * (ip_float + g_k1xsumq);

            unsafe {
                *output.get_unchecked_mut(i) = (vec_idx, est_dist);
            }
        }

        output
    }
// EVOLVE-BLOCK-END
