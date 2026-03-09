// This file is spliced into source_template.rs by evaluator.py.
// It intentionally includes the explicit AVX2 FastScan helper path plus the
// oracle methods that consume it.

// EVOLVE-AVX2-BLOCK-START
    /// Explicitly call the AVX2 implementation (the target of this benchmark).
    /// Panics if AVX2 is not available.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    pub fn fast_scan_avx2<const B: usize>(packed_codes: &[u8], luts: &[u8], results: &mut [u16]) {
        avx2_impl::fast_scan_avx2::<B>(packed_codes, luts, results);
    }

    /// AVX2-optimized implementation.
    /// Always compiled on x86_64 with AVX2 for benchmarking comparisons.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    mod avx2_impl {
        use std::arch::x86_64::*;

        /// Unsafe AVX2 kernel to accumulate distances for one block of 32 vectors.
        ///
        /// This is a literal port of the original `accumulate_one_block`
        /// routine from the C++ implementation.
        #[target_feature(enable = "avx2")]
        unsafe fn accumulate_one_block<const B: usize>(
            codes: *const u8,
            lut: *const u8,
            result: *mut u16,
        ) {
            let m = B / 4;
            let low_mask = _mm256_set1_epi8(0x0F);
            let mut accu0 = _mm256_setzero_si256();
            let mut accu1 = _mm256_setzero_si256();
            let mut accu2 = _mm256_setzero_si256();
            let mut accu3 = _mm256_setzero_si256();

            let mut codes_ptr = codes as *const __m256i;
            let mut lut_ptr = lut as *const __m256i;

            for _ in 0..(m / 2) {
                let c = _mm256_loadu_si256(codes_ptr);
                let lo = _mm256_and_si256(c, low_mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

                let lut_m = _mm256_loadu_si256(lut_ptr);
                let res_lo = _mm256_shuffle_epi8(lut_m, lo);
                let res_hi = _mm256_shuffle_epi8(lut_m, hi);

                accu0 = _mm256_add_epi16(accu0, res_lo);
                accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
                accu2 = _mm256_add_epi16(accu2, res_hi);
                accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

                codes_ptr = codes_ptr.add(1);
                lut_ptr = lut_ptr.add(1);
            }

            accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
            let dis0 = _mm256_add_epi16(
                _mm256_permute2f128_si256(accu0, accu1, 0x21),
                _mm256_blend_epi32(accu0, accu1, 0xF0),
            );
            _mm256_storeu_si256(result as *mut __m256i, dis0);

            accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
            let dis1 = _mm256_add_epi16(
                _mm256_permute2f128_si256(accu2, accu3, 0x21),
                _mm256_blend_epi32(accu2, accu3, 0xF0),
            );
            _mm256_storeu_si256(result.add(16) as *mut __m256i, dis1);
        }

        pub fn fast_scan_avx2<const B: usize>(
            packed_codes: &[u8],
            luts: &[u8],
            results: &mut [u16],
        ) {
            let m = B / 4;
            let n_vecs = results.len();
            if n_vecs == 0 {
                return;
            }
            assert_eq!(
                n_vecs % 32,
                0,
                "Vector count must be a multiple of 32 for AVX2 FastScan"
            );

            let n_blocks = n_vecs / 32;
            let block_size_bytes = (m / 2) * 32;

            assert_eq!(
                packed_codes.len(),
                n_blocks * block_size_bytes,
                "Packed codes length is incorrect"
            );
            assert_eq!(luts.len(), m * 16, "LUTs length is incorrect");

            let codes_ptr = packed_codes.as_ptr();
            let luts_ptr = luts.as_ptr();
            let results_ptr = results.as_mut_ptr();

            for i in 0..n_blocks {
                unsafe {
                    accumulate_one_block::<B>(
                        codes_ptr.add(i * block_size_bytes),
                        luts_ptr,
                        results_ptr.add(i * 32),
                    )
                }
            }
        }
    }
// EVOLVE-AVX2-BLOCK-END

// EVOLVE-ORACLE-BLOCK-START
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

        unsafe {
            let codes_ptr = self.fastscan.packed_codes.as_ptr().add(code_start);
            let codes_slice = std::slice::from_raw_parts(codes_ptr, block_size_bytes);

            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                rabitq_fast_scan_kernels::fast_scan_avx2::<B>(codes_slice, &self.luts, &mut results);
            }

            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                rabitq_fast_scan_kernels::fast_scan_dispatch::<B>(
                    codes_slice,
                    &self.luts,
                    &mut results,
                );
            }
        }

        results
    }

    /// Computes approximate L2 distances for a single block.
    ///
    /// Uses the RaBitQ distance estimation formula:
    /// `est_dist = f_add + g_add + f_rescale * (ip_result + g_k1xsumq)`
    ///
    /// Returns 32 (index, distance) pairs.
    #[fastrace::trace(name = "compare")]
    pub fn compute_block_distances(&self, block_idx: usize) -> [(usize, f32); BLOCK_SIZE] {
        let raw_results = self.compute_block_raw(block_idx);
        let base_idx = block_idx * BLOCK_SIZE;

        let mut output = [(0usize, 0.0f32); BLOCK_SIZE];

        let g_add = self.g_add;
        let g_k1xsumq = self.g_k1xsumq;
        let lut_delta = self.lut_delta;
        let lut_sum_vl = self.lut_sum_vl;

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
            let ip_float = lut_delta * raw_ip + lut_sum_vl;
            let factors = unsafe { self.fastscan.factors.get_unchecked(vec_idx) };
            let est_dist = factors.f_add + g_add + factors.f_rescale * (ip_float + g_k1xsumq);

            unsafe {
                *output.get_unchecked_mut(i) = (vec_idx, est_dist);
            }
        }

        output
    }
// EVOLVE-ORACLE-BLOCK-END
