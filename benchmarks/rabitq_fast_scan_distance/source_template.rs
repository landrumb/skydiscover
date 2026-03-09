//! Implementation of the RaBitQ "Fast Scan" logic using AVX2 intrinsics.
//!
//! This module provides functions to:
//! 1. `pack_lut`: Create the 16-entry Look-Up Tables (LUTs) from a quantized query vector.
//! 2. `fast_scan_dispatch`: Perform the high-speed, batched distance accumulation,
//!    dispatching to an AVX2 kernel when available.
//! 3. `RabitqFastScan`: A struct that stores packed binary codes and can produce
//!    `DistanceOracle` instances for efficient block-wise distance computation.
//!
//! The logic is based on the C++ implementation in `src/fast_scan.h` [cite: gaoj0017/rabitq/RaBitQ-785450bae8b8ad9c5025159f5f70270e92f7084e/src/fast_scan.h]
//! and the accompanying technical report [cite: 383-387].

use crate::data_handling::dataset::VectorDataset;
use crate::data_handling::dataset_traits::DistanceOracle;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Contains the public dispatch function for SIMD-based "FAST_SCAN".
pub mod rabitq_fast_scan_kernels {
    // This `pos` array is used to build the LUT. It maps the bit position
    // in a 4-bit code to the corresponding index in the 4-byte query chunk.
    // Based on `fast_scan.h` [cite: gaoj0017/rabitq/RaBitQ-785450bae8b8ad9c5025159f5f70270e92f7084e/src/fast_scan.h].
    const POS: [usize; 16] = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3];
    const BLOCK_WIDTH: usize = 32;
    const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

    /// Creates the Look-Up Tables (LUTs) from a quantized query.
    ///
    /// * `quantized_query`: A `B`-byte slice where each `u8` is a quantized
    ///   value (0-15).
    /// * `lut_output`: A mutable slice to write the LUTs into. Must be
    ///   `B / 4 * 16` bytes long.
    ///
    /// This is a Rust port of the `pack_LUT` function in `src/fast_scan.h` [cite: gaoj0017/rabitq/RaBitQ-785450bae8b8ad9c5025159f5f70270e92f7084e/src/fast_scan.h].
    pub fn pack_lut<const B: usize>(quantized_query: &[u8], lut_output: &mut [u8]) {
        let m = B / 4;
        assert_eq!(quantized_query.len(), B, "Quantized query length must be B");
        assert_eq!(lut_output.len(), m * 16, "LUT output length must be M * 16");

        let query_chunks = quantized_query.chunks_exact(4);
        let lut_chunks = lut_output.chunks_exact_mut(16);

        // Process 4 query bytes (one M-segment) at a time to create one 16-byte LUT
        for (bq, lut) in query_chunks.zip(lut_chunks) {
            lut[0] = 0;
            for j in 1..16 {
                // `j & (-j)` is a bit-hack for `lowbit(j)`
                let lowbit = j & (-(j as i32) as usize);
                lut[j] = lut[j - lowbit] + bq[POS[j]];
            }
        }
    }

    /// Packs the binary RaBitQ codes (stored as `u64` bit-strings) into the
    /// Faiss-style FastScan layout used by the AVX2 kernel.
    ///
    /// The key insight is that the LUT is built such that for nibble value j:
    /// - bit 0 of j → query position 3 in the 4-element chunk
    /// - bit 1 of j → query position 2
    /// - bit 2 of j → query position 1
    /// - bit 3 of j → query position 0
    ///
    /// So binary code bits at positions [4m, 4m+1, 4m+2, 4m+3] must be packed as:
    /// nibble_m = bit[4m+3] | (bit[4m+2] << 1) | (bit[4m+1] << 2) | (bit[4m+0] << 3)
    pub fn pack_codes_from_binary<const B: usize>(binary_codes: &[u64]) -> Vec<u8> {
        assert!(B.is_multiple_of(64), "B ({}) must be divisible by 64", B);
        let n_u64 = B / 64;
        assert_eq!(
            binary_codes.len() % n_u64,
            0,
            "Binary codes length must be a multiple of B/64"
        );

        let n_vecs = binary_codes.len() / n_u64;
        if n_vecs == 0 {
            return Vec::new();
        }
        assert_eq!(
            n_vecs % BLOCK_WIDTH,
            0,
            "FastScan requires the vector count to be a multiple of 32"
        );

        let m = B / 4; // Number of nibbles per vector
        let bytes_per_vec = m / 2; // Each byte holds 2 nibbles

        // Step 1: Convert binary codes to nibbles with correct bit ordering
        // For each group of 4 bits at positions [4m, 4m+1, 4m+2, 4m+3],
        // create nibble with bits reversed: bit[4m+3] | (bit[4m+2]<<1) | (bit[4m+1]<<2) | (bit[4m+0]<<3)
        let mut byte_codes = vec![0u8; n_vecs * bytes_per_vec];

        for vec_idx in 0..n_vecs {
            let code_base = vec_idx * n_u64;

            for nibble_idx in 0..m {
                // Get the 4 bits for this nibble
                let bit_base = nibble_idx * 4;
                let word_idx = bit_base / 64;
                let bit_offset = bit_base % 64;

                let word = binary_codes[code_base + word_idx];

                // Extract bits at positions bit_base, bit_base+1, bit_base+2, bit_base+3
                // Handle the case where bits span two u64 words
                let bits = if bit_offset <= 60 {
                    // All 4 bits are in the same word
                    ((word >> bit_offset) & 0xF) as u8
                } else {
                    // Bits span two words (this only happens when bit_offset > 60)
                    let bits_in_first = 64 - bit_offset;
                    let first_part = (word >> bit_offset) as u8;
                    let second_word = binary_codes[code_base + word_idx + 1];
                    let second_part = (second_word << bits_in_first) as u8;
                    (first_part | second_part) & 0xF
                };

                // Reverse the bit order within the nibble:
                // bit 0 of nibble <- bit 3 of binary (position 4m+3)
                // bit 1 of nibble <- bit 2 of binary (position 4m+2)
                // bit 2 of nibble <- bit 1 of binary (position 4m+1)
                // bit 3 of nibble <- bit 0 of binary (position 4m+0)
                let reversed_nibble =
                    ((bits & 1) << 3) | ((bits & 2) << 1) | ((bits & 4) >> 1) | ((bits & 8) >> 3);

                // Store nibble in the byte array (2 nibbles per byte)
                // Match C++ convention: upper nibble = even-indexed nibble (dims 0-3),
                // lower nibble = odd-indexed nibble (dims 4-7)
                let byte_idx = nibble_idx / 2;
                let nibble_pos = nibble_idx % 2;
                if nibble_pos == 0 {
                    // Even nibble goes in HIGH 4 bits (like C++ col_0 = byte >> 4)
                    byte_codes[vec_idx * bytes_per_vec + byte_idx] |= reversed_nibble << 4;
                } else {
                    // Odd nibble goes in LOW 4 bits (like C++ col_1 = byte & 15)
                    byte_codes[vec_idx * bytes_per_vec + byte_idx] |= reversed_nibble;
                }
            }
        }

        // Step 2: Pack blocks exactly like the C++ implementation
        let mut packed = vec![0u8; n_vecs * bytes_per_vec];
        let mut dest_offset = 0;
        let mut column = [0u8; BLOCK_WIDTH];
        let mut c0 = [0u8; BLOCK_WIDTH];
        let mut c1 = [0u8; BLOCK_WIDTH];

        for block_start in (0..n_vecs).step_by(BLOCK_WIDTH) {
            for m_idx in (0..m).step_by(2) {
                let col_idx = m_idx / 2;
                for (lane, column_item) in column.iter_mut().enumerate().take(BLOCK_WIDTH) {
                    let row = block_start + lane;
                    *column_item = byte_codes[row * bytes_per_vec + col_idx];
                }

                for lane in 0..BLOCK_WIDTH {
                    // Match C++ convention: col_0 = byte >> 4 (high nibble = even-indexed),
                    // col_1 = byte & 15 (low nibble = odd-indexed)
                    c0[lane] = column[lane] >> 4;
                    c1[lane] = column[lane] & 0x0F;
                }

                for j in 0..16 {
                    let idx = PERM0[j];
                    packed[dest_offset + j] = c0[idx] | (c0[idx + 16] << 4);
                    packed[dest_offset + j + 16] = c1[idx] | (c1[idx + 16] << 4);
                }
                dest_offset += BLOCK_WIDTH;
            }
        }

        packed
    }

    /// Inverse of `pack_codes_from_binary`. This can be used to recover
    /// the `u64` bit-string layout from packed FastScan blocks so that
    /// scalar and SIMD paths operate on identical datasets.
    pub fn unpack_packed_codes<const B: usize>(packed_codes: &[u8]) -> Vec<u64> {
        let m = B / 4;
        let bytes_per_vec = m / 2;
        let block_size_bytes = bytes_per_vec * BLOCK_WIDTH;
        assert_eq!(
            packed_codes.len() % block_size_bytes,
            0,
            "Packed codes length must be a multiple of block size"
        );

        let n_blocks = packed_codes.len() / block_size_bytes;
        let n_vecs = n_blocks * BLOCK_WIDTH;
        let n_u64 = B / 64;

        // Step 1: Unpack the block layout back to per-vector nibble bytes
        let mut codes_bytes = vec![0u8; n_vecs * bytes_per_vec];

        for block_idx in 0..n_blocks {
            for col in 0..bytes_per_vec {
                let chunk = &packed_codes[block_idx * block_size_bytes + col * BLOCK_WIDTH
                    ..block_idx * block_size_bytes + (col + 1) * BLOCK_WIDTH];
                let mut c0 = [0u8; BLOCK_WIDTH];
                let mut c1 = [0u8; BLOCK_WIDTH];
                for j in 0..16 {
                    let d0 = chunk[j];
                    let d1 = chunk[j + 16];
                    let idx = PERM0[j];
                    c0[idx] = d0 & 0x0F;
                    c0[idx + 16] = d0 >> 4;
                    c1[idx] = d1 & 0x0F;
                    c1[idx + 16] = d1 >> 4;
                }
                for lane in 0..BLOCK_WIDTH {
                    let row = block_idx * BLOCK_WIDTH + lane;
                    // Match C++ convention: high nibble = c0 (even-indexed),
                    // low nibble = c1 (odd-indexed)
                    codes_bytes[row * bytes_per_vec + col] = (c0[lane] << 4) | c1[lane];
                }
            }
        }

        // Step 2: Convert nibbles back to binary codes
        // Reverse the bit ordering within each nibble and reassemble u64s
        let mut binary_codes = vec![0u64; n_vecs * n_u64];

        for vec_idx in 0..n_vecs {
            let code_base = vec_idx * n_u64;

            for nibble_idx in 0..m {
                // Get the nibble from byte_codes
                // Match C++ convention: upper nibble = even-indexed nibble,
                // lower nibble = odd-indexed nibble
                let byte_idx = nibble_idx / 2;
                let nibble_pos = nibble_idx % 2;
                let nibble = if nibble_pos == 0 {
                    // Even nibble is in HIGH 4 bits
                    codes_bytes[vec_idx * bytes_per_vec + byte_idx] >> 4
                } else {
                    // Odd nibble is in LOW 4 bits
                    codes_bytes[vec_idx * bytes_per_vec + byte_idx] & 0x0F
                };

                // Reverse the bit order (undo the reversal done in packing)
                let original_bits = ((nibble & 1) << 3)
                    | ((nibble & 2) << 1)
                    | ((nibble & 4) >> 1)
                    | ((nibble & 8) >> 3);

                // Store bits back into the binary code
                let bit_base = nibble_idx * 4;
                let word_idx = bit_base / 64;
                let bit_offset = bit_base % 64;

                if bit_offset <= 60 {
                    // All 4 bits go in the same word
                    binary_codes[code_base + word_idx] |= (original_bits as u64) << bit_offset;
                } else {
                    // Bits span two words
                    let bits_in_first = 64 - bit_offset;
                    binary_codes[code_base + word_idx] |= (original_bits as u64) << bit_offset;
                    binary_codes[code_base + word_idx + 1] |=
                        (original_bits as u64) >> bits_in_first;
                }
            }
        }

        binary_codes
    }

    /// Scalar reference implementation of FastScan used for validation.
    pub fn fast_scan_reference<const B: usize>(
        packed_codes: &[u8],
        luts: &[u8],
        results: &mut [u16],
    ) {
        let m = B / 4;
        let block_size_bytes = (m / 2) * BLOCK_WIDTH;
        let n_blocks = packed_codes.len() / block_size_bytes;
        assert_eq!(
            packed_codes.len() % block_size_bytes,
            0,
            "Packed codes length must align with block size"
        );
        assert_eq!(
            results.len(),
            n_blocks * BLOCK_WIDTH,
            "Results length must match number of vectors"
        );
        assert_eq!(luts.len(), m * 16, "LUT array must have M * 16 entries");

        results.fill(0);

        for block_idx in 0..n_blocks {
            let block_codes =
                &packed_codes[block_idx * block_size_bytes..(block_idx + 1) * block_size_bytes];
            let block_results =
                &mut results[block_idx * BLOCK_WIDTH..(block_idx + 1) * BLOCK_WIDTH];

            for pair in 0..(m / 2) {
                let chunk = &block_codes[pair * BLOCK_WIDTH..(pair + 1) * BLOCK_WIDTH];
                let lut_chunk = &luts[pair * 32..(pair + 1) * 32];
                let (lut0, lut1) = lut_chunk.split_at(16);

                let mut c0 = [0u8; BLOCK_WIDTH];
                let mut c1 = [0u8; BLOCK_WIDTH];
                for j in 0..16 {
                    let d0 = chunk[j];
                    let d1 = chunk[j + 16];
                    let idx = PERM0[j];
                    c0[idx] = d0 & 0x0F;
                    c0[idx + 16] = d0 >> 4;
                    c1[idx] = d1 & 0x0F;
                    c1[idx + 16] = d1 >> 4;
                }

                for vec in 0..BLOCK_WIDTH {
                    block_results[vec] += lut0[c0[vec] as usize] as u16;
                    block_results[vec] += lut1[c1[vec] as usize] as u16;
                }
            }
        }
    }

    /// Public dispatch function for batched distance accumulation.
    ///
    /// This function will call the AVX512/AVX2-optimized kernel if the hardware
    /// supports it, otherwise it falls back to the scalar implementation.
    ///
    /// * `packed_codes`: The database codes, packed in 32-vector blocks
    ///   using the Faiss-style layout.
    /// * `luts`: The pre-computed LUTs from `pack_lut`.
    /// * `results`: A mutable slice to store the `u16` accumulated distances.
    ///   Length must be a multiple of 32.
    #[inline]
    pub fn fast_scan_dispatch<const B: usize>(
        packed_codes: &[u8],
        luts: &[u8],
        results: &mut [u16],
    ) {
        // Prefer AVX-512 when available (best performance)
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        {
            avx512_impl::fast_scan_avx512::<B>(packed_codes, luts, results);
        }

        // Fall back to AVX2 if AVX-512 is not available
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        {
            avx2_impl::fast_scan_avx2::<B>(packed_codes, luts, results);
        }

        // Fall back to scalar implementation if neither AVX-512 nor AVX2 is available
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            fast_scan_reference::<B>(packed_codes, luts, results);
        }
    }

    /// Explicitly call the AVX-512 implementation (for benchmarking).
    /// Panics if AVX-512 is not available.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[inline]
    pub fn fast_scan_avx512<const B: usize>(packed_codes: &[u8], luts: &[u8], results: &mut [u16]) {
        avx512_impl::fast_scan_avx512::<B>(packed_codes, luts, results);
    }

    __EVOLVE_AVX2_BLOCK__

    /// Explicitly call the scalar reference implementation (for benchmarking).
    #[inline]
    pub fn fast_scan_scalar<const B: usize>(packed_codes: &[u8], luts: &[u8], results: &mut [u16]) {
        fast_scan_reference::<B>(packed_codes, luts, results);
    }

    /// AVX-512 optimized implementation (best performance).
    /// Processes 64 bytes per iteration vs 32 bytes for AVX2.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    mod avx512_impl {
        use std::arch::x86_64::*;

        /// Unsafe AVX-512 kernel to accumulate distances for one block of 32 vectors.
        ///
        /// This processes the entire block in one pass using 512-bit registers,
        /// achieving ~2x throughput compared to AVX2 for the inner loop.
        ///
        /// The key insight is that both `codes` and `lut` are accessed with the
        /// same offset and have the same total size (B * 4 bytes). The layout is:
        /// - codes[0..31]: packed codes for sub-quantizers 0-1 (32 vectors)
        /// - codes[32..63]: packed codes for sub-quantizers 2-3 (32 vectors)
        /// - ... and so on
        /// - lut[0..15]: LUT for sub-quantizer 0
        /// - lut[16..31]: LUT for sub-quantizer 1
        /// - ... and so on
        ///
        /// AVX-512 loads 64 bytes at a time, processing 4 sub-quantizers per iteration.
        #[target_feature(enable = "avx512f", enable = "avx512bw")]
        unsafe fn accumulate_one_block_avx512<const B: usize>(
            codes: *const u8,
            lut: *const u8,
            result: *mut u16,
        ) {
            // code_length = dim * 4 = B * 4 bytes (same as LUT size)
            // In terms of m: code_length = (m/2) * 32 = m * 16
            let code_length = B * 4;

            let lo_mask = _mm512_set1_epi8(0x0F);
            let mut accu0 = _mm512_setzero_si512();
            let mut accu1 = _mm512_setzero_si512();
            let mut accu2 = _mm512_setzero_si512();
            let mut accu3 = _mm512_setzero_si512();

            // Process 64 bytes per iteration
            // The C++ code: for (size_t i = 0; i < code_length; i += 64)
            let mut i = 0;
            while i < code_length {
                // Load 64 bytes of packed codes and LUTs
                // Both are accessed with the same offset (key design of FastScan)
                let c = _mm512_loadu_si512(codes.add(i) as *const __m512i);
                let lut_m = _mm512_loadu_si512(lut.add(i) as *const __m512i);

                // Extract low nibbles (vectors 0-15) and high nibbles (vectors 16-31)
                let lo = _mm512_and_si512(c, lo_mask);
                let hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

                // Perform 128 parallel table lookups using vpshufb
                let res_lo = _mm512_shuffle_epi8(lut_m, lo);
                let res_hi = _mm512_shuffle_epi8(lut_m, hi);

                // Accumulate results as i16 to avoid overflow
                // Due to the interleaved data order (0, 8, 1, 9, 2, 10, ...):
                // - accu0/accu1 handle vectors 0-15 (even/odd bytes)
                // - accu2/accu3 handle vectors 16-31 (even/odd bytes)
                accu0 = _mm512_add_epi16(accu0, res_lo);
                accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
                accu2 = _mm512_add_epi16(accu2, res_hi);
                accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));

                i += 64;
            }

            // Remove the influence of upper 8 bits for accu0 and accu2
            accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
            accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

            // Combine accumulators from the 4 x 128-bit lanes
            // ret1 = results for vectors 0-15
            let ret1 = _mm512_add_epi16(
                _mm512_mask_blend_epi64(0b11110000, accu0, accu1),
                _mm512_shuffle_i64x2(accu0, accu1, 0b01001110),
            );
            // ret2 = results for vectors 16-31
            let ret2 = _mm512_add_epi16(
                _mm512_mask_blend_epi64(0b11110000, accu2, accu3),
                _mm512_shuffle_i64x2(accu2, accu3, 0b01001110),
            );

            // Final combination: merge the two halves to get all 32 results
            let mut ret = _mm512_setzero_si512();
            ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b10001000));
            ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));

            // Store all 32 u16 results in one 64-byte store
            _mm512_storeu_si512(result as *mut __m512i, ret);
        }

        /// Public AVX-512 function.
        #[inline]
        pub fn fast_scan_avx512<const B: usize>(
            packed_codes: &[u8],
            luts: &[u8],
            results: &mut [u16],
        ) {
            let m = B / 4;
            let n_vecs = results.len();
            if n_vecs == 0 {
                return;
            }
            debug_assert_eq!(
                n_vecs % 32,
                0,
                "Vector count must be a multiple of 32 for AVX512 FastScan"
            );

            let n_blocks = n_vecs / 32;
            let block_size_bytes = (m / 2) * 32; // = m * 16 = B * 4 bytes per block

            debug_assert_eq!(
                packed_codes.len(),
                n_blocks * block_size_bytes,
                "Packed codes length is incorrect"
            );
            // LUT size should equal block_size_bytes (both are B * 4)
            debug_assert_eq!(luts.len(), m * 16, "LUTs length is incorrect");

            let codes_ptr = packed_codes.as_ptr();
            let luts_ptr = luts.as_ptr();
            let results_ptr = results.as_mut_ptr();

            for i in 0..n_blocks {
                unsafe {
                    accumulate_one_block_avx512::<B>(
                        codes_ptr.add(i * block_size_bytes),
                        luts_ptr,
                        results_ptr.add(i * 32),
                    )
                }
            }
        }
    }

    // Note: The scalar fallback uses fast_scan_reference directly via fast_scan_scalar.
}

/// Contains the public dispatch function for scalar "SCAN".
/// This is a port of the `SCAN` path from the C++ implementation,
/// which uses bitwise operations on `u64` blocks.
/// [cite: gaoj0017/rabitq/RaBitQ-785450bae8b8ad9c5025159f5f70270e92f7084e/src/space.h]
/// [cite: gaoj0017/rabitq/RaBitQ-785450bae8b8ad9c5025159f5f70270e92f7084e/src/ivf_rabitq.h]
pub mod rabitq_scalar_scan {

    /// Rust port of `space.h::ip_bin_bin` [cite: gaoj0017/rabitq/RaBitQ-785450bae8b8ad9c5025159f5f70270e92f7084e/src/space.h].
    /// Computes the inner product (popcount of AND) between two binary vectors.
    #[inline]
    fn ip_bin_bin<const B: usize>(q: &[u64], d: &[u64]) -> u32 {
        let n_u64 = B / 64;
        assert_eq!(q.len(), n_u64);
        assert_eq!(d.len(), n_u64);

        let mut ret: u32 = 0;
        for i in 0..n_u64 {
            ret += (q[i] & d[i]).count_ones();
        }
        ret
    }

    /// Rust port of `space.h::ip_byte_bin` [cite: gaoj0017/rabitq/RaBitQ-785450bae8b8ad9c5025159f5f70270e92f7084e/src/space.h].
    /// Computes the inner product <xb, qu> using bit-slicing.
    #[inline]
    pub fn ip_byte_bin<const B: usize, const B_QUERY: usize>(
        transposed_query: &[u64],
        data_code: &[u64],
    ) -> u32 {
        let n_u64 = B / 64;
        assert_eq!(data_code.len(), n_u64);
        assert_eq!(transposed_query.len(), B_QUERY * n_u64);

        let mut ret: u32 = 0;
        for i in 0..B_QUERY {
            let q_chunk = &transposed_query[i * n_u64..(i + 1) * n_u64];
            ret += ip_bin_bin::<B>(q_chunk, data_code) << i;
        }
        ret
    }

    /// Rust port of `space.h::transpose_bin` [cite: gaoj0017/rabitq/RaBitQ-785450bae8b8ad9c5025159f5f70270e92f7084e/src/space.h].
    /// This is a scalar implementation, not the SIMD one from the C++ file.
    /// It converts the `B`-byte quantized query (0-15) into a
    /// bit-sliced layout for `ip_byte_bin`.
    pub fn transpose_bin<const B: usize, const B_QUERY: usize>(
        quantized_query: &[u8],
        transposed_query: &mut [u64],
    ) {
        let n_u64 = B / 64;
        assert_eq!(quantized_query.len(), B);
        assert_eq!(transposed_query.len(), B_QUERY * n_u64);

        transposed_query.fill(0);

        for (i, &val_u8) in quantized_query.iter().enumerate().take(B) {
            let val = val_u8 as u64; // 0-15
            for j in 0..B_QUERY {
                if (val & (1 << j)) != 0 {
                    let tq_idx = j * n_u64 + (i / 64);
                    let bit_pos = i % 64;
                    transposed_query[tq_idx] |= 1 << bit_pos;
                }
            }
        }
    }

    /// Public dispatch function for scalar distance accumulation.
    ///
    /// * `binary_codes`: The database codes in their raw `u64` bit-string layout.
    /// * `quantized_query`: The `B`-byte quantized query (values 0-15).
    /// * `results`: A mutable slice to store the `u32` accumulated distances.
    pub fn scalar_scan_dispatch<const B: usize, const B_QUERY: usize>(
        binary_codes: &[u64],
        quantized_query: &[u8],
        results: &mut [u32],
    ) {
        let n_u64 = B / 64;
        let n_vecs = results.len();
        if n_vecs == 0 {
            return;
        }
        assert_eq!(binary_codes.len(), n_vecs * n_u64);
        assert_eq!(quantized_query.len(), B);

        // --- Query Preprocessing ---
        let mut transposed_query = vec![0u64; B_QUERY * n_u64];
        transpose_bin::<B, B_QUERY>(quantized_query, &mut transposed_query);
        // --- End Query Preprocessing ---

        for i in 0..n_vecs {
            let data_chunk = &binary_codes[i * n_u64..(i + 1) * n_u64];
            results[i] = ip_byte_bin::<B, B_QUERY>(&transposed_query, data_chunk);
        }
    }

    /// Variant of `scalar_scan_dispatch` that processes data in arbitrary
    /// block orders of 32 vectors (matching the FAST_SCAN block layout).
    ///
    /// * `block_indices`: A list of block indices (each block represents
    ///   32 vectors) dictating the processing order. The length of `results`
    ///   must be `block_indices.len() * 32`.
    pub fn scalar_scan_dispatch_block_order<const B: usize, const B_QUERY: usize>(
        binary_codes: &[u64],
        quantized_query: &[u8],
        block_indices: &[usize],
        results: &mut [u32],
    ) {
        const BLOCK_SIZE: usize = 32;

        if block_indices.is_empty() {
            return;
        }
        let n_u64 = B / 64;
        let block_stride = BLOCK_SIZE * n_u64;

        assert_eq!(
            results.len(),
            block_indices.len() * BLOCK_SIZE,
            "Results length must equal block_indices.len() * 32"
        );
        assert_eq!(quantized_query.len(), B, "Quantized query length must be B");
        assert_eq!(
            binary_codes.len() % block_stride,
            0,
            "Binary codes length must be a multiple of 32 vectors"
        );

        let total_blocks = binary_codes.len() / block_stride;
        for &block_idx in block_indices {
            assert!(
                block_idx < total_blocks,
                "Block index {} out of bounds (total blocks {})",
                block_idx,
                total_blocks
            );
        }

        let mut transposed_query = vec![0u64; B_QUERY * n_u64];
        transpose_bin::<B, B_QUERY>(quantized_query, &mut transposed_query);

        for (block_pos, &block_idx) in block_indices.iter().enumerate() {
            let data_start = block_idx * block_stride;
            let block_data = &binary_codes[data_start..data_start + block_stride];
            let result_offset = block_pos * BLOCK_SIZE;
            for lane in 0..BLOCK_SIZE {
                let data_chunk = &block_data[lane * n_u64..(lane + 1) * n_u64];
                results[result_offset + lane] =
                    ip_byte_bin::<B, B_QUERY>(&transposed_query, data_chunk);
            }
        }
    }
}

/// Block size for FastScan operations (always 32 vectors per block).
pub const BLOCK_SIZE: usize = 32;

/// Per-vector factors used to convert raw FastScan inner product results
/// into approximate Euclidean distances.
///
/// For RaBitQ 1-bit quantization, the distance estimate is:
/// `est_dist = f_add + ||q||² + f_rescale * (inner_product_result)`
///
/// where:
/// - `f_add = ||x-c||² + 2*||x-c||² * <c, x̄_cb> / <x-c, x̄_cb>`
/// - `f_rescale = -2 * ||x-c||² / <x-c, x̄_cb>`
/// - `x̄_cb` is the centered binary code (binary_code - 0.5)
#[derive(Clone, Debug)]
pub struct VectorFactors {
    /// Factor added to the distance estimate: f_add = ||x-c||² * (1 + 2*<c, x̄_cb>/<x-c, x̄_cb>)
    pub f_add: f32,
    /// Scale factor for inner product: f_rescale = -2*||x-c||²/<x-c, x̄_cb>
    pub f_rescale: f32,
    /// Error bound factor for distance estimation.
    pub f_error: f32,
}

impl Default for VectorFactors {
    fn default() -> Self {
        Self {
            f_add: 0.0,
            f_rescale: 0.0,
            f_error: 0.0,
        }
    }
}

/// RaBitQ FastScan index for efficient block-wise distance computation.
///
/// This struct stores:
/// - Packed binary codes in the Faiss-style FastScan layout
/// - Per-vector metadata for distance reconstruction
///
/// The generic parameter `B` is the dimension (must be a multiple of 64).
pub struct RabitqFastScan<const B: usize> {
    /// Number of actual vectors in the dataset.
    n: usize,
    /// Number of 32-vector blocks (includes padding).
    n_blocks: usize,
    /// Packed codes in FastScan layout.
    packed_codes: Vec<u8>,
    /// Binary codes in u64 bit-string layout (for scalar fallback).
    binary_codes: Vec<u64>,
    /// Per-vector factors for distance reconstruction.
    factors: Vec<VectorFactors>,
}

impl<const B: usize> RabitqFastScan<B> {
    /// Creates a new `RabitqFastScan` from binary codes and per-vector factors.
    ///
    /// # Arguments
    /// * `binary_codes` - Binary codes in u64 bit-string layout. Length must be
    ///   `n_vecs * (B/64)` where `n_vecs` is a multiple of 32.
    /// * `factors` - Per-vector factors. Length must equal `n_vecs`.
    ///
    /// # Panics
    /// Panics if `B` is not a multiple of 64, or if the input lengths are inconsistent.
    #[allow(clippy::manual_is_multiple_of)]
    pub fn from_binary_codes(binary_codes: Vec<u64>, factors: Vec<VectorFactors>) -> Self {
        assert!(B.is_multiple_of(64), "B ({}) must be divisible by 64", B);
        let n_u64 = B / 64;

        assert_eq!(
            binary_codes.len() % n_u64,
            0,
            "Binary codes length must be a multiple of B/64"
        );
        let n = binary_codes.len() / n_u64;
        assert_eq!(
            factors.len(),
            n,
            "Factors length ({}) must match vector count ({})",
            factors.len(),
            n
        );
        assert_eq!(
            n % BLOCK_SIZE,
            0,
            "Vector count ({}) must be a multiple of {} for FastScan",
            n,
            BLOCK_SIZE
        );

        let n_blocks = n / BLOCK_SIZE;
        let packed_codes = rabitq_fast_scan_kernels::pack_codes_from_binary::<B>(&binary_codes);

        Self {
            n,
            n_blocks,
            packed_codes,
            binary_codes,
            factors,
        }
    }

    /// Creates a new `RabitqFastScan` from a float vector dataset.
    ///
    /// This extracts the sign bits of each dimension as the binary code and
    /// computes the squared norm of each vector.
    ///
    /// # Arguments
    /// * `dataset` - A float vector dataset with dimension `B`.
    ///
    /// # Panics
    /// Panics if the dataset dimension doesn't match `B`, or if the vector count
    /// is not a multiple of 32 (use `from_f32_dataset_padded` for automatic padding).
    pub fn from_f32_dataset(dataset: &VectorDataset<f32>) -> Self {
        assert_eq!(
            dataset.dim, B,
            "Dataset dimension ({}) must match B ({})",
            dataset.dim, B
        );
        if !dataset.n.is_multiple_of(BLOCK_SIZE) {
            eprintln!("Warning: Dataset size ({}) is not a multiple of {} for FastScan. Padding to the next multiple of {}.", dataset.n, BLOCK_SIZE, BLOCK_SIZE);
        }

        let padded_n = dataset.n.div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
        Self::build_from_f32_internal(dataset, padded_n)
    }

    /// Creates a new `RabitqFastScan` from a float vector dataset, padding to
    /// the next multiple of 32 vectors if necessary.
    ///
    /// Padded vectors are filled with zeros and will have zero norm.
    pub fn from_f32_dataset_padded(dataset: &VectorDataset<f32>) -> Self {
        assert_eq!(
            dataset.dim, B,
            "Dataset dimension ({}) must match B ({})",
            dataset.dim, B
        );

        let padded_n = dataset.n.div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
        Self::build_from_f32_internal(dataset, padded_n)
    }

    fn build_from_f32_internal(dataset: &VectorDataset<f32>, target_n: usize) -> Self {
        let n_u64 = B / 64;
        let actual_n = dataset.n;

        let mut binary_codes = vec![0u64; target_n * n_u64];
        let mut factors = vec![VectorFactors::default(); target_n];

        // RaBitQ uses centroid = 0 for simplicity (global quantization)
        // c_b = -(2^1 - 1) / 2 = -0.5 (centering constant for 1-bit code)
        const C_B: f32 = -0.5;
        const EPSILON: f32 = 1.9; // Error bound constant from RaBitQ paper

        for i in 0..actual_n {
            let vec = dataset.get(i);

            // Compute binary code (sign bits), squared norm, and RaBitQ factors
            let mut norm_sq = 0.0f32;
            let mut ip_vec_xucb = 0.0f32; // <vec, centered_code>

            for (d, &val) in vec.iter().enumerate().take(B) {
                norm_sq += val * val;

                // Set bit if positive (binary_code[d] = 1 if val > 0, else 0)
                let bit = if val > 0.0 { 1.0f32 } else { 0.0f32 };
                if val > 0.0 {
                    let word_idx = d / 64;
                    let bit_idx = d % 64;
                    binary_codes[i * n_u64 + word_idx] |= 1u64 << bit_idx;
                }

                // Centered code: xu_cb[d] = binary_code[d] + c_b = binary_code[d] - 0.5
                let xu_cb_d = bit + C_B;
                ip_vec_xucb += val * xu_cb_d;
            }

            // Compute RaBitQ factors for L2 distance estimation
            // With centroid = 0:
            // - f_add = ||x||² (since ip_cent_xucb = 0)
            // - f_rescale = -2 * ||x||² / <x, xu_cb>
            // Corner case: if ip_vec_xucb ≈ 0, set to infinity to avoid division issues
            let ip_vec_xucb = if ip_vec_xucb.abs() < 1e-10 {
                f32::INFINITY
            } else {
                ip_vec_xucb
            };

            let l2_norm = norm_sq.sqrt();
            let xu_cb_norm_sq = B as f32 * 0.25; // ||xu_cb||² = B * 0.25 for centered 0/1 codes

            // Error factor (simplified from C++ implementation)
            let tmp_error = if ip_vec_xucb.is_finite() {
                let ratio = (norm_sq * xu_cb_norm_sq) / (ip_vec_xucb * ip_vec_xucb);
                if ratio > 1.0 && B > 1 {
                    l2_norm * EPSILON * ((ratio - 1.0) / (B as f32 - 1.0)).sqrt()
                } else {
                    0.0
                }
            } else {
                0.0
            };

            factors[i] = VectorFactors {
                f_add: norm_sq, // For centroid=0: f_add = ||x||²
                f_rescale: -2.0 * norm_sq / ip_vec_xucb,
                f_error: 2.0 * tmp_error,
            };
        }

        let n_blocks = target_n / BLOCK_SIZE;
        let packed_codes = rabitq_fast_scan_kernels::pack_codes_from_binary::<B>(&binary_codes);

        Self {
            n: actual_n,
            n_blocks,
            packed_codes,
            binary_codes,
            factors,
        }
    }

    /// Returns the number of vectors in the dataset (excluding padding).
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the number of 32-vector blocks.
    pub fn num_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Returns the per-vector factors.
    pub fn factors(&self) -> &[VectorFactors] {
        &self.factors
    }

    /// Returns the packed codes (for direct AVX2 operations).
    pub fn packed_codes(&self) -> &[u8] {
        &self.packed_codes
    }

    /// Returns the binary codes in u64 layout (for scalar operations).
    pub fn binary_codes(&self) -> &[u64] {
        &self.binary_codes
    }

    /// Creates a `DistanceOracle` for the given query vector.
    ///
    /// The oracle computes approximate squared Euclidean distances using the
    /// FastScan inner product kernel.
    ///
    /// # Arguments
    /// * `query` - Query vector of dimension `B`.
    ///
    /// # Returns
    /// A `RabitqFastScanOracle` that implements `DistanceOracle`.
    pub fn make_oracle<'a>(&'a self, query: &[f32]) -> RabitqFastScanOracle<'a, B> {
        RabitqFastScanOracle::new(self, query)
    }

    /// Returns the bytes per block for the packed codes layout.
    fn block_size_bytes(&self) -> usize {
        let m = B / 4;
        (m / 2) * BLOCK_SIZE
    }

    /// Saves the FastScan index to disk (binary format).
    ///
    /// File layout (little endian):
    /// magic "RBFS" | version:u32 | dim:u32 | n:u64 | n_blocks:u64 |
    /// packed_len:u64 | packed_codes | binary_len:u64 | binary_codes |
    /// factors_len:u64 | factors[f_add,f_rescale,f_error]*.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        writer.write_all(b"RBFS")?;
        writer.write_all(&1u32.to_le_bytes())?;
        writer.write_all(&(B as u32).to_le_bytes())?;
        writer.write_all(&(self.n as u64).to_le_bytes())?;
        writer.write_all(&(self.n_blocks as u64).to_le_bytes())?;

        writer.write_all(&(self.packed_codes.len() as u64).to_le_bytes())?;
        writer.write_all(&self.packed_codes)?;

        writer.write_all(&(self.binary_codes.len() as u64).to_le_bytes())?;
        for &word in &self.binary_codes {
            writer.write_all(&word.to_le_bytes())?;
        }

        writer.write_all(&(self.factors.len() as u64).to_le_bytes())?;
        for f in &self.factors {
            writer.write_all(&f.f_add.to_le_bytes())?;
            writer.write_all(&f.f_rescale.to_le_bytes())?;
            writer.write_all(&f.f_error.to_le_bytes())?;
        }

        writer.flush()
    }

    /// Loads a FastScan index from disk written by [`save`].
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"RBFS" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid RaBitQ FastScan file (bad magic)",
            ));
        }

        let version = read_u32(&mut reader)?;
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported RaBitQ FastScan version {}", version),
            ));
        }

        let dim = read_u32(&mut reader)? as usize;
        if dim != B {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Dimension mismatch: file={} expected={}", dim, B),
            ));
        }

        let n = read_u64(&mut reader)? as usize;
        let n_blocks = read_u64(&mut reader)? as usize;

        let packed_len = read_u64(&mut reader)? as usize;
        let mut packed_codes = vec![0u8; packed_len];
        reader.read_exact(&mut packed_codes)?;

        let binary_len = read_u64(&mut reader)? as usize;
        let mut binary_codes = vec![0u64; binary_len];
        for word in &mut binary_codes {
            *word = read_u64(&mut reader)?;
        }

        let factors_len = read_u64(&mut reader)? as usize;
        let mut factors = Vec::with_capacity(factors_len);
        for _ in 0..factors_len {
            let f_add = read_f32(&mut reader)?;
            let f_rescale = read_f32(&mut reader)?;
            let f_error = read_f32(&mut reader)?;
            factors.push(VectorFactors {
                f_add,
                f_rescale,
                f_error,
            });
        }

        // Validate lengths
        let expected_target_n = n_blocks * BLOCK_SIZE;
        let n_u64 = B / 64;
        let m = B / 4;
        let expected_packed_len = (m / 2) * BLOCK_SIZE * n_blocks;
        let expected_binary_len = expected_target_n * n_u64;

        if packed_len != expected_packed_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Packed codes length mismatch: file={} expected={}",
                    packed_len, expected_packed_len
                ),
            ));
        }
        if binary_len != expected_binary_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Binary codes length mismatch: file={} expected={}",
                    binary_len, expected_binary_len
                ),
            ));
        }
        if factors_len != expected_target_n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Factors length mismatch: file={} expected={}",
                    factors_len, expected_target_n
                ),
            ));
        }

        Ok(Self {
            n,
            n_blocks,
            packed_codes,
            binary_codes,
            factors,
        })
    }
}

#[inline]
fn read_u32<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[inline]
fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[inline]
fn read_f32<R: Read>(reader: &mut R) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// A query-specific oracle for RaBitQ FastScan distance computation.
///
/// This implements `DistanceOracle` where `compare(block_idx)` returns
/// distances for all 32 vectors in the specified block.
///
/// The distance estimation formula for L2 is:
/// `est_dist = f_add + g_add + f_rescale * (ip_result + g_k1xsumq)`
///
/// where:
/// - `f_add`, `f_rescale` are per-vector factors from quantization
/// - `g_add = ||q||²` (query squared norm)
/// - `g_k1xsumq = c_b * sum(q)` where `c_b = -0.5`
/// - `ip_result = <query, code>` (inner product from FastScan)
#[derive(Clone)]
pub struct RabitqFastScanOracle<'a, const B: usize> {
    /// Reference to the parent FastScan index.
    fastscan: &'a RabitqFastScan<B>,
    /// Precomputed LUTs from float query values (quantized to u8).
    luts: Vec<u8>,
    /// Query-side additive factor: g_add = ||q||²
    g_add: f32,
    /// Query-side inner product offset: g_k1xsumq = c_b * sum(q) = -0.5 * sum(q)
    g_k1xsumq: f32,
    /// LUT dequantization scale: delta = lut_range / 255
    lut_delta: f32,
    /// LUT dequantization offset: sum_vl = lut_min * (B/4)
    lut_sum_vl: f32,
}

impl<const B: usize> crate::util::Named for RabitqFastScanOracle<'_, B> {
    fn name(&self) -> &str {
        "RabitqFastScanOracle"
    }
}

impl<'a, const B: usize> RabitqFastScanOracle<'a, B> {
    /// Creates a new oracle from a FastScan index and query vector.
    ///
    /// The LUT is built from actual float query values (not binary).
    /// FastScan then computes: `<query, code>` where code[i] ∈ {0, 1}
    pub fn new(fastscan: &'a RabitqFastScan<B>, query: &[f32]) -> Self {
        assert_eq!(
            query.len(),
            B,
            "Query dimension ({}) must match B ({})",
            query.len(),
            B
        );

        // Compute query-side factors
        // g_add = ||q||² (query squared norm)
        let g_add: f32 = query.iter().map(|x| x * x).sum();

        // g_k1xsumq = c_b * sum(q) where c_b = -0.5 (centering constant)
        let sum_q: f32 = query.iter().sum();
        let g_k1xsumq = -0.5 * sum_q;

        // Build LUT from actual float query values
        // The LUT entry for nibble j is: sum of query[i] for each bit i set in j
        let m = B / 4;
        let mut float_lut = vec![0.0f32; m * 16];

        // Build float LUT using the same algorithm as pack_lut
        const POS: [usize; 16] = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3];
        for (chunk_idx, lut_chunk) in float_lut.chunks_exact_mut(16).enumerate() {
            let query_chunk = &query[chunk_idx * 4..(chunk_idx + 1) * 4];
            lut_chunk[0] = 0.0;
            for j in 1..16 {
                let lowbit = j & (-(j as i32) as usize);
                lut_chunk[j] = lut_chunk[j - lowbit] + query_chunk[POS[j]];
            }
        }

        // Find min/max for scalar quantization
        let lut_min = float_lut.iter().cloned().fold(f32::INFINITY, f32::min);
        let lut_max = float_lut.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lut_range = lut_max - lut_min;

        // Compute the maximum quantization level that won't overflow u16 when summed.
        // The FastScan kernel sums m LUT entries (one per sub-quantizer), so:
        // max_sum = m * max_quant_level <= u16::MAX
        // => max_quant_level <= u16::MAX / m
        // We use a slightly conservative value to avoid edge cases.
        let max_quant_level = (u16::MAX as usize / m).min(255) as u8;

        // Compute dequantization parameters (precomputed for fast distance computation)
        // delta = lut_range / max_quant_level
        // sum_vl = lut_min * num_tables (baseline when all quantized values are 0)
        let lut_delta = if lut_range > 1e-10 {
            lut_range / (max_quant_level as f32)
        } else {
            0.0
        };
        let lut_sum_vl = lut_min * (m as f32);

        // Quantize LUT to u8 (values 0 to max_quant_level)
        let mut luts = vec![0u8; m * 16];
        let scale = if lut_range > 1e-10 {
            (max_quant_level as f32) / lut_range
        } else {
            1.0
        };
        for (i, &fval) in float_lut.iter().enumerate() {
            let quantized = ((fval - lut_min) * scale).round() as i32;
            luts[i] = quantized.clamp(0, max_quant_level as i32) as u8;
        }

        Self {
            fastscan,
            luts,
            g_add,
            g_k1xsumq,
            lut_delta,
            lut_sum_vl,
        }
    }

    /// Returns the LUTs.
    pub fn luts(&self) -> &[u8] {
        &self.luts
    }

    /// Returns a reference to the parent FastScan index.
    pub fn fastscan(&self) -> &RabitqFastScan<B> {
        self.fastscan
    }

    __EVOLVE_ORACLE_BLOCK__
}

impl<const B: usize> DistanceOracle for RabitqFastScanOracle<'_, B> {
    /// Computes distances for a block of 32 vectors.
    ///
    /// # Arguments
    /// * `block_idx` - The block index (0-indexed). Each block contains 32 vectors.
    ///
    /// # Returns
    /// A boxed slice of (vector_index, distance) pairs for all 32 vectors in the block.
    /// For padding vectors beyond the actual dataset size, the distance is computed
    /// but callers should filter by index if needed.
    #[fastrace::trace(name = "compare")]
    fn compare(&self, block_idx: usize) -> Box<[(usize, f32)]> {
        let distances = self.compute_block_distances(block_idx);
        distances.to_vec().into_boxed_slice()
    }
}

/// A query-specific oracle for RaBitQ scalar distance computation.
///
/// Unlike `RabitqFastScanOracle` which computes distances in blocks of 32,
/// this oracle computes distances one vector at a time using direct LUT lookups.
/// This matches the FastScan distance formula but without SIMD batching.
#[derive(Clone)]
pub struct RabitqScalarOracle<'a, const B: usize> {
    /// Reference to the parent FastScan index (for binary codes and factors).
    fastscan: &'a RabitqFastScan<B>,
    /// Float LUT values for distance computation (m * 16 entries).
    float_lut: Vec<f32>,
    /// Query-side additive factor: g_add = ||q||²
    pub g_add: f32,
    /// Query-side multiplicative term: g_k1xsumq = c_b * sum(q)
    pub g_k1xsumq: f32,
}

impl<const B: usize> crate::util::Named for RabitqScalarOracle<'_, B> {
    fn name(&self) -> &str {
        "RabitqScalarOracle"
    }
}

impl<'a, const B: usize> RabitqScalarOracle<'a, B> {
    /// Creates a new scalar oracle from a FastScan index and query vector.
    ///
    /// Uses the same LUT-based approach as FastScan but computes distances
    /// one vector at a time instead of in blocks.
    ///
    /// # Arguments
    /// * `fastscan` - The RaBitQ FastScan index containing binary codes and factors
    /// * `query` - Query vector of dimension B
    pub fn new(fastscan: &'a RabitqFastScan<B>, query: &[f32]) -> Self {
        assert_eq!(query.len(), B, "Query dimension must match B");

        // Compute query-side factors (same as FastScan oracle)
        let g_add: f32 = query.iter().map(|x| x * x).sum();
        let sum_q: f32 = query.iter().sum();
        let g_k1xsumq = -0.5 * sum_q;

        // Build float LUT using the same algorithm as FastScan
        // The LUT entry for nibble j is: sum of query[i] for each bit i set in j
        let m = B / 4;
        let mut float_lut = vec![0.0f32; m * 16];

        const POS: [usize; 16] = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3];
        for (chunk_idx, lut_chunk) in float_lut.chunks_exact_mut(16).enumerate() {
            let query_chunk = &query[chunk_idx * 4..(chunk_idx + 1) * 4];
            lut_chunk[0] = 0.0;
            for j in 1..16 {
                let lowbit = j & (-(j as i32) as usize);
                lut_chunk[j] = lut_chunk[j - lowbit] + query_chunk[POS[j]];
            }
        }

        Self {
            fastscan,
            float_lut,
            g_add,
            g_k1xsumq,
        }
    }

    /// Compute RaBitQ distance for a single vector using LUT lookups.
    ///
    /// This implements the same distance formula as FastScan but without SIMD:
    /// `est_dist = f_add + g_add + f_rescale * (ip_result + g_k1xsumq)`
    ///
    /// Uses binary_codes (u64 bit-string layout) which is simpler to access
    /// than the complex packed layout used by SIMD.
    ///
    /// # Arguments
    /// * `idx` - Vector index in the dataset
    ///
    /// # Returns
    /// Estimated squared Euclidean distance
    #[fastrace::trace(name = "compare")]
    pub fn compute_distance(&self, idx: usize) -> f32 {
        let binary_codes = self.fastscan.binary_codes();
        let factors = self.fastscan.factors();
        let m = B / 4; // Number of sub-quantizers (nibbles)
        let n_u64 = B / 64;

        // Get the binary code for this vector
        let code_start = idx * n_u64;

        // Accumulate inner product by looking up LUT entries for each sub-quantizer
        let mut ip_sum: f32 = 0.0;

        for sq in 0..m {
            // Each sub-quantizer covers 4 consecutive bits
            let bit_base = sq * 4;
            let word_idx = bit_base / 64;
            let bit_offset = bit_base % 64;

            let word = binary_codes[code_start + word_idx];

            // Extract the 4 bits for this sub-quantizer
            let bits = if bit_offset <= 60 {
                ((word >> bit_offset) & 0xF) as u8
            } else {
                // Bits span two words
                let bits_in_first = 64 - bit_offset;
                let first_part = (word >> bit_offset) as u8;
                let second_word = binary_codes[code_start + word_idx + 1];
                let second_part = (second_word << bits_in_first) as u8;
                (first_part | second_part) & 0xF
            };

            // Reverse the bit order within the nibble to match pack_codes_from_binary:
            // The LUT was built with the reversed bit order convention
            // bit 0 of nibble <- bit 3 of binary
            // bit 1 of nibble <- bit 2 of binary
            // bit 2 of nibble <- bit 1 of binary
            // bit 3 of nibble <- bit 0 of binary
            let reversed_nibble =
                ((bits & 1) << 3) | ((bits & 2) << 1) | ((bits & 4) >> 1) | ((bits & 8) >> 3);

            // Look up LUT value
            ip_sum += self.float_lut[sq * 16 + reversed_nibble as usize];
        }

        // Apply distance estimation formula
        let factor = &factors[idx];
        let distance = factor.f_add + self.g_add + factor.f_rescale * (ip_sum + self.g_k1xsumq);

        // Increment distance comparison counter if dcmp feature is enabled
        #[cfg(feature = "dcmp")]
        {
            use crate::distance::increment_distance_comparison_count;
            increment_distance_comparison_count();
        }

        distance
    }

    /// Returns a reference to the parent FastScan index.
    pub fn fastscan(&self) -> &RabitqFastScan<B> {
        self.fastscan
    }

    /// Maximum valid vector index (for filtering padding vectors).
    pub fn max_valid_idx(&self) -> usize {
        self.fastscan().len()
    }
}

// ============================================================================
// BeamSearchOracle implementations for RaBitQ oracles
// ============================================================================

use crate::data_handling::beam_search_oracle::BeamSearchOracle;
use crate::graph::{IndexT, SeenSet};
use std::ops::Range;

impl<const B: usize> BeamSearchOracle for RabitqFastScanOracle<'_, B> {
    fn initialize(
        &self,
        start: IndexT,
        seen: &mut SeenSet,
        scanned_blocks: &mut SeenSet,
    ) -> (f32, Vec<(IndexT, f32)>, usize) {
        let max_valid = self.fastscan().len();
        let start_block = start as usize / BLOCK_SIZE;
        let block_distances = self.compute_block_distances(start_block);

        // Find the distance for the start node
        let initial_distance = block_distances
            .iter()
            .find(|(idx, _)| *idx == start as usize)
            .map(|(_, d)| *d)
            .unwrap_or(f32::INFINITY);

        // Mark the start block as scanned
        scanned_blocks.insert(start_block as IndexT);

        // Collect additional distances from the start block
        let additional: Vec<_> = block_distances
            .iter()
            .filter(|&&(idx, _)| {
                idx < max_valid && idx != start as usize && seen.insert(idx as IndexT)
            })
            .map(|&(idx, dist)| (idx as IndexT, dist))
            .collect();

        (initial_distance, additional, BLOCK_SIZE)
    }

    fn process_neighbors<'a, I: Iterator<Item = &'a IndexT>>(
        &self,
        neighbors: I,
        seen: &mut SeenSet,
        scanned_blocks: &mut SeenSet,
        frontier: &mut Vec<(IndexT, f32)>,
        cutoff: Option<f32>,
    ) -> usize {
        let max_valid = self.fastscan().len();
        let mut comparisons = 0;

        // Collect unique block indices for neighbors that haven't been scanned
        let mut blocks_to_scan: Vec<usize> = neighbors
            .map(|&neighbor| neighbor as usize / BLOCK_SIZE)
            .filter(|block_idx| !scanned_blocks.contains(&(*block_idx as IndexT)))
            .collect();
        blocks_to_scan.sort_unstable();
        blocks_to_scan.dedup();

        // Use cutoff if provided, otherwise accept all candidates
        let cutoff_dist = cutoff.unwrap_or(f32::INFINITY);

        // Reserve capacity upfront to avoid repeated reallocations
        frontier.reserve(blocks_to_scan.len() * BLOCK_SIZE);

        // Scan each new block and add all distances to the frontier
        for block_idx in blocks_to_scan {
            if scanned_blocks.insert(block_idx as IndexT) {
                let block_distances = self.compute_block_distances(block_idx);
                comparisons += BLOCK_SIZE;

                // Add all valid vectors from this block using extend (faster than individual pushes)
                frontier.extend(
                    block_distances
                        .into_iter()
                        .filter(|&(idx, dist)| {
                            idx < max_valid && seen.insert(idx as IndexT) && dist < cutoff_dist
                        })
                        .map(|(idx, dist)| (idx as IndexT, dist)),
                );
            }
        }

        comparisons
    }

    fn process_range(
        &self,
        range: Range<IndexT>,
        seen: &mut SeenSet,
        scanned_blocks: &mut SeenSet,
        frontier: &mut Vec<(IndexT, f32)>,
        cutoff: Option<f32>,
    ) -> usize {
        if range.is_empty() {
            return 0;
        }

        // Use cutoff if provided, otherwise accept all candidates
        let cutoff_dist = cutoff.unwrap_or(f32::INFINITY);

        // we use the endpoints of the range to efficiently identify the relevant blocks
        let start_block = range.start as usize / BLOCK_SIZE;
        let end_block = (range.end as usize - 1) / BLOCK_SIZE;
        let mut comparisons = 0;
        for block_idx in start_block..=end_block {
            if scanned_blocks.insert(block_idx as IndexT) {
                let block_distances = self.compute_block_distances(block_idx);
                comparisons += BLOCK_SIZE;
                frontier.extend(
                    block_distances
                        .into_iter()
                        .filter(|&(idx, dist)| seen.insert(idx as IndexT) && dist < cutoff_dist)
                        .map(|(idx, dist)| (idx as IndexT, dist)),
                );
            }
        }
        comparisons
    }

    fn compute_distance(&self, idx: usize) -> f32 {
        let block_idx = idx / BLOCK_SIZE;
        let block_distances = self.compute_block_distances(block_idx);
        block_distances
            .iter()
            .find(|(i, _)| *i == idx)
            .map(|&(_, d)| d)
            .unwrap_or(f32::INFINITY)
    }

    fn max_valid_idx(&self) -> usize {
        self.fastscan().len()
    }

    fn num_blocks(&self) -> usize {
        self.fastscan().num_blocks()
    }
}

impl<const B: usize> BeamSearchOracle for RabitqScalarOracle<'_, B> {
    fn initialize(
        &self,
        start: IndexT,
        _seen: &mut SeenSet,
        _scanned_blocks: &mut SeenSet,
    ) -> (f32, Vec<(IndexT, f32)>, usize) {
        let initial_distance = self.compute_distance(start as usize);
        // Scalar oracle only computes the start distance, no additional vectors
        (initial_distance, Vec::new(), 1)
    }

    fn process_neighbors<'a, I: Iterator<Item = &'a IndexT>>(
        &self,
        neighbors: I,
        seen: &mut SeenSet,
        _scanned_blocks: &mut SeenSet,
        frontier: &mut Vec<(IndexT, f32)>,
        cutoff: Option<f32>,
    ) -> usize {
        let max_valid = self.max_valid_idx();
        let cutoff_dist = cutoff.unwrap_or(f32::INFINITY);
        let mut comparisons = 0;

        for &neighbor in neighbors {
            if neighbor as usize >= max_valid {
                continue; // Skip padding
            }

            if seen.insert(neighbor) {
                let dist = self.compute_distance(neighbor as usize);
                comparisons += 1;
                if dist < cutoff_dist {
                    frontier.push((neighbor, dist));
                }
            }
        }

        comparisons
    }

    fn process_range(
        &self,
        range: Range<IndexT>,
        seen: &mut SeenSet,
        _scanned_blocks: &mut SeenSet,
        frontier: &mut Vec<(IndexT, f32)>,
        cutoff: Option<f32>,
    ) -> usize {
        self.process_neighbors(
            range.into_iter().collect::<Vec<_>>().iter(),
            seen,
            _scanned_blocks,
            frontier,
            cutoff,
        )
    }

    fn compute_distance(&self, idx: usize) -> f32 {
        RabitqScalarOracle::compute_distance(self, idx)
    }

    fn max_valid_idx(&self) -> usize {
        self.fastscan().len()
    }
}
