//! Implementation of a regular degree-limited graph

use std::fs::File;
use std::io::{BufReader, Read, Write};

use crate::graph::VectorGraph;

use super::{Graph, IndexT, MutableGraph, SeenSet, SeenSetImpl};
use crate::data_handling::dataset_traits::DistanceOracle;
use crate::graph::{Beam, BeamImpl, BeamSearchable, SIZE_HINT_FACTOR};
use std::sync::Arc;

#[cfg(feature = "metrics")]
use metrics::counter;

pub struct ClassicGraph {
    edges: Box<[IndexT]>,
    pub n: IndexT, // number of nodes
    pub r: usize,  // degree limit
}

impl crate::util::Named for ClassicGraph {
    fn name(&self) -> &str {
        "ClassicGraph"
    }
}

impl Graph for ClassicGraph {
    fn neighbors(&self, i: IndexT) -> &[IndexT] {
        self.get_neighborhood(i)
    }
}

impl MutableGraph for ClassicGraph {
    fn add_neighbor(&mut self, from: IndexT, to: IndexT) {
        self.add_edge(from, to);
    }

    fn set_neighborhood(&mut self, i: IndexT, neighborhood: &[IndexT]) {
        self.set_neighborhood(i, neighborhood);
    }
}

impl ClassicGraph {
    pub fn new(n: IndexT, r: usize) -> ClassicGraph {
        let entries_per_node = r + 1;
        let edges = vec![0; n as usize * entries_per_node].into_boxed_slice();

        ClassicGraph { edges, n, r }
    }

    #[inline]
    fn offset(&self, i: IndexT) -> usize {
        (i as usize) * (self.r + 1)
    }

    #[inline]
    fn degree_at(&self, i: IndexT) -> usize {
        self.edges[self.offset(i)] as usize
    }

    /// save the graph to a file
    /// uses the parlayANN format for compatibility
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        // println!(
        //     "Writing graph with {} nodes and max degree {}",
        //     self.n, self.r
        // );

        // Write header: n and r (as u32)
        file.write_all(&self.n.to_le_bytes())?;
        file.write_all(&(self.r as u32).to_le_bytes())?;

        // Write degrees for each node
        for i in 0..self.n {
            let deg = self.degree_at(i) as IndexT;
            file.write_all(&deg.to_le_bytes())?;
        }

        // Write edge data sequentially for each node in blocks (similar to the C++ impl)
        const BLOCK_SIZE: usize = 1_000_000;
        let mut node_index = 0;

        while node_index < self.n as usize {
            let block_start = node_index;
            let block_end = (block_start + BLOCK_SIZE).min(self.n as usize);

            // Gather all edges for this block
            let mut block_edges = Vec::new();
            for i in block_start..block_end {
                let degree = self.degree_at(i as IndexT);
                let offset = self.offset(i as IndexT);
                block_edges.extend_from_slice(&self.edges[offset + 1..offset + 1 + degree]);
            }

            // Write all edges in this block
            for &edge in &block_edges {
                file.write_all(&edge.to_le_bytes())?;
            }

            node_index = block_end;
        }

        Ok(())
    }

    /// read a graph from a file in parlayANN format
    pub fn read(path: &str) -> std::io::Result<ClassicGraph> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header: n and r
        let mut header_buf = [0u8; 8];
        reader.read_exact(&mut header_buf)?;
        let n = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
        let r = u32::from_le_bytes(header_buf[4..8].try_into().unwrap()) as usize;

        // println!("Reading graph with {} nodes and max degree {}", n, r);

        // Read all degrees at once
        let mut degrees = vec![0u32; n as usize];
        for degree in degrees.iter_mut() {
            let mut degree_buf = [0u8; 4];
            reader.read_exact(&mut degree_buf)?;
            *degree = u32::from_le_bytes(degree_buf);
        }

        // Calculate total edges
        // let total_edges: usize = degrees.iter().map(|&d| d as usize).sum();
        // println!("Total edges in graph: {}", total_edges);

        let entries_per_node = r + 1;
        let mut edges = vec![0u32; n as usize * entries_per_node];

        // Read edges in blocks (like the C++ implementation)
        const BLOCK_SIZE: usize = 1_000_000;
        let mut node_index = 0;

        while node_index < n as usize {
            let block_start = node_index;
            let block_end = (block_start + BLOCK_SIZE).min(n as usize);

            // Calculate how many edges to read for this block
            let block_edges: usize = degrees[block_start..block_end]
                .iter()
                .map(|&d| d as usize)
                .sum();

            // Read all edges for this block
            let mut edges_buf = vec![0u8; block_edges * 4];
            reader.read_exact(&mut edges_buf)?;

            // Distribute edges to neighborhoods
            let mut edge_offset = 0;
            for (i, &deg) in degrees[block_start..block_end].iter().enumerate() {
                let degree = deg as usize;
                let offset = (block_start + i) * entries_per_node;
                edges[offset] = degree as IndexT;
                for j in 0..degree {
                    let start_byte = (edge_offset + j) * 4;
                    let end_byte = start_byte + 4;
                    edges[offset + 1 + j] =
                        u32::from_le_bytes(edges_buf[start_byte..end_byte].try_into().unwrap());
                }
                edge_offset += degree;
            }

            node_index = block_end;
        }

        Ok(ClassicGraph {
            edges: edges.into_boxed_slice(),
            n,
            r,
        })
    }

    /// returns a view of the neighborhood of a node
    pub fn get_neighborhood(&self, i: IndexT) -> &[IndexT] {
        assert!(i < self.n);
        let offset = self.offset(i);
        let degree = self.edges[offset] as usize;
        &self.edges[offset + 1..offset + 1 + degree]
    }

    /// overwrites the neighborhood of a node
    pub fn set_neighborhood(&mut self, i: IndexT, neighborhood: &[IndexT]) {
        assert!(i < self.n);
        assert!(neighborhood.len() <= self.r); // neighborhood must be smaller than the degree limit
        let offset = self.offset(i);
        self.edges[offset] = neighborhood.len() as IndexT;
        self.edges[offset + 1..offset + 1 + neighborhood.len()].copy_from_slice(neighborhood);
    }

    /// adds a neighbor to a node
    pub fn add_edge(&mut self, from: IndexT, to: IndexT) {
        assert!(from < self.n && to < self.n);
        let offset = self.offset(from);
        let degree = self.edges[offset] as usize;
        assert!(degree < self.r);

        self.edges[offset + 1 + degree] = to;
        self.edges[offset] += 1;
    }

    /// adds an undirected edge (adds in both directions)
    pub fn add_undirected_edge(&mut self, a: IndexT, b: IndexT) {
        self.add_edge(a, b);
        self.add_edge(b, a);
    }

    /// returns the number of nodes in the graph
    pub fn size(&self) -> usize {
        self.n as usize
    }

    /// returns the maximum degree of the graph
    pub fn max_degree(&self) -> usize {
        self.r
    }

    /// returns the current degree of a node
    pub fn degree(&self, i: IndexT) -> usize {
        assert!(i < self.n);
        self.degree_at(i)
    }

    /// clears the neighborhood of a node
    pub fn clear_neighborhood(&mut self, i: IndexT) {
        assert!(i < self.n);
        let offset = self.offset(i);
        self.edges[offset] = 0;
    }

    /// appends a list of neighbors to a node
    pub fn append_neighbors(&mut self, i: IndexT, neighbors: &[IndexT]) {
        assert!(i < self.n);
        let offset = self.offset(i);
        let current_degree = self.edges[offset] as usize;
        assert!(
            current_degree + neighbors.len() <= self.r,
            "Cannot exceed max degree {} for node {}",
            self.r,
            i
        );

        let start = offset + 1 + current_degree;
        let end = start + neighbors.len();
        self.edges[start..end].copy_from_slice(neighbors);
        self.edges[offset] += neighbors.len() as IndexT;
    }

    /// sorts the neighborhood of a node according to a comparator
    pub fn sort_neighborhood<F>(&mut self, i: IndexT, comparator: F)
    where
        F: FnMut(&IndexT, &IndexT) -> std::cmp::Ordering,
    {
        assert!(i < self.n);
        let offset = self.offset(i);
        let degree = self.edges[offset] as usize;
        self.edges[offset + 1..offset + 1 + degree].sort_by(comparator);
    }

    /// returns an EdgeRange view of the neighborhood of a node
    /// this is equivalent to the [] operator in the C++ implementation
    pub fn get_edge_range(&self, i: IndexT) -> EdgeRange<'_> {
        assert!(i < self.n, "graph index out of range: {i}");
        EdgeRange::new(self.get_neighborhood(i), i)
    }

    /// returns the practical memory footprint of the graph in bytes
    pub fn naive_footprint(&self) -> usize {
        self.n as usize * (self.r + 1) * std::mem::size_of::<IndexT>()
    }

    /// returns the memory footprint of a CSR-based representation of the graph.
    pub fn efficient_footprint(&self) -> usize {
        let total_degree: usize = (0..self.n).map(|i| self.degree_at(i)).sum();
        (self.n as usize + total_degree) * std::mem::size_of::<IndexT>()
    }
}

impl std::ops::Index<IndexT> for ClassicGraph {
    type Output = [IndexT];

    fn index(&self, index: IndexT) -> &Self::Output {
        self.get_neighborhood(index)
    }
}

/// A wrapper around a slice of neighbors, providing similar functionality to the C++ edgeRange
pub struct EdgeRange<'a> {
    neighbors: &'a [IndexT],
    id: IndexT,
}

impl<'a> EdgeRange<'a> {
    pub fn new(neighbors: &'a [IndexT], id: IndexT) -> Self {
        EdgeRange { neighbors, id }
    }

    pub fn size(&self) -> usize {
        self.neighbors.len()
    }

    pub fn id(&self) -> IndexT {
        self.id
    }

    /// prefetch the neighborhood into cache (similar to C++ implementation)
    pub fn prefetch(&self) {
        // This is a no-op in Rust since we don't have direct cache control
        // The C++ version uses __builtin_prefetch
    }
}

impl From<ClassicGraph> for VectorGraph {
    fn from(graph: ClassicGraph) -> VectorGraph {
        let mut neighborhoods = Vec::new();
        for i in 0..graph.n {
            neighborhoods.push(graph.get_neighborhood(i).to_vec());
        }
        VectorGraph::new(neighborhoods)
    }
}

/// Internal state for performing beam search on a `ClassicGraph` using the `BeamSearchable` trait
#[derive(Clone)]
pub struct ClassicBeamState {
    /// set of nodes that have been seen (to avoid re-inserting)
    seen: SeenSet,
    /// beam manager for frontier and visited tracking
    beam: Beam,
    /// the width of the beam
    beam_width: usize,
    /// optional limit on number of expansions
    limit: Option<usize>,
    /// oracle used to compute distances
    oracle: Arc<dyn DistanceOracle>,
    /// optional number of seen vectors to keep for reranking
    rerank_n: Option<usize>,
    #[cfg(feature = "record_beams")]
    /// snapshots of the beam (frontier ids) after each expansion
    beam_snapshots: Vec<Vec<IndexT>>,
}

impl ClassicBeamState {
    /// Create a new search state given an oracle, starting node, beam width, and optional expansion limit.
    pub fn new(
        oracle: Arc<dyn DistanceOracle>,
        start: IndexT,
        beam_width: usize,
        limit: Option<usize>,
        initial_distance: f32,
        graph_size: usize,
    ) -> Self {
        let mut seen = SeenSet::new(graph_size, (beam_width + 10) * SIZE_HINT_FACTOR);
        seen.insert(start);
        let beam_length = beam_width.max(1);
        let mut beam = Beam::new(beam_length, graph_size);
        beam.insert((start, initial_distance));
        ClassicBeamState {
            seen,
            beam,
            beam_width,
            limit,
            oracle,
            rerank_n: None,
            #[cfg(feature = "record_beams")]
            beam_snapshots: Vec::new(),
        }
    }
}

impl ClassicBeamState {
    #[cfg(feature = "record_beams")]
    pub fn get_beam_snapshots(&self) -> &Vec<Vec<IndexT>> {
        &self.beam_snapshots
    }
}

impl ClassicGraph {
    /// Instrumented helper for expanding neighbors during beam search.
    #[fastrace::trace(name = "expand_neighborhood")]
    fn expand_neighborhood(&self, state: &mut ClassicBeamState, current: IndexT) {
        // Explore neighbors and add unseen ones to the frontier with computed distances
        let neighbors = self.neighbors(current);
        #[cfg(feature = "metrics")]
        let total_candidates = neighbors.len();
        // Collect neighbors that haven't been seen yet
        let unseen_neighbors: Vec<IndexT> = neighbors
            .iter()
            .copied()
            .filter(|&neighbor| state.seen.insert(neighbor))
            .collect();

        #[cfg(feature = "metrics")]
        {
            let skipped = total_candidates.saturating_sub(unseen_neighbors.len());
            if skipped > 0 {
                counter!("neighborhood_skipped_comparisons").increment(skipped as u64);
            }
        }

        // Batch compare all unseen neighbors at once
        if !unseen_neighbors.is_empty() {
            #[cfg(feature = "metrics")]
            {
                counter!("expand_neighborhood_comparisons_total")
                    .increment(unseen_neighbors.len() as u64);
            }
            let batch_results = state.oracle.compare_batch(&unseen_neighbors);
            for &(idx, dist) in batch_results.iter() {
                if state.seen.insert(idx as IndexT) || unseen_neighbors.contains(&(idx as IndexT)) {
                    state.beam.insert((idx as IndexT, dist));
                }
            }
        }
    }
}

impl BeamSearchable for ClassicGraph {
    type Node = IndexT;
    type Neighbor = IndexT;
    type SearchState = ClassicBeamState;

    #[fastrace::trace(name = "next_node")]
    fn next_node(&self, search_state: &mut Self::SearchState) -> Option<Self::Node> {
        if let Some(limit) = search_state.limit {
            if search_state.beam.visited_count() >= limit {
                return None;
            }
        }
        search_state.beam.get_next()
    }

    #[fastrace::trace(name = "expand_node")]
    fn expand_node(&self, state: &mut Self::SearchState, node: Self::Node) {
        #[cfg(feature = "metrics")]
        {
            counter!("expand_node_calls_total").increment(1);
        }
        let current = node;
        self.expand_neighborhood(state, current);

        #[cfg(feature = "record_beams")]
        {
            // record beam snapshot (ids only) at end of expansion
            let _ = state.beam.get_results();
            let snapshot: Vec<IndexT> = state.beam.beam().iter().map(|(id, _)| *id).collect();
            state.beam_snapshots.push(snapshot);
        }
    }

    #[fastrace::trace(name = "get_results")]
    fn get_results(&self, state: &mut Self::SearchState) -> Vec<IndexT> {
        let result_count = state.rerank_n.unwrap_or(state.beam_width);
        state
            .beam
            .get_results()
            .into_iter()
            .take(result_count)
            .collect()
    }
}

impl std::ops::Index<usize> for EdgeRange<'_> {
    type Output = IndexT;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.neighbors.len(),
            "index exceeds degree while accessing neighbors"
        );
        &self.neighbors[index]
    }
}

// ============================================================================
// RaBitQ FastScan-based beam search implementation
// ============================================================================

use crate::data_handling::rabitq_fast_scan::{
    RabitqFastScan, RabitqFastScanOracle, RabitqScalarOracle,
};
use crate::data_handling::BeamSearchOracle;

/// A wrapper around `ClassicGraph` for oracle-based beam search.
///
/// This generic wrapper allows beam search using different oracle types:
/// - `RabitqFastScanOracle<B>`: Block-based distance computation (32 vectors at a time)
/// - `RabitqScalarOracle<B>`: One-at-a-time distance computation
///
/// The oracle type `O` determines the search behavior through the `BeamSearchable` trait implementation.
/// The actual oracle instance is stored in the search state, not in this wrapper,
/// because the oracle is query-specific.
pub struct ClassicGraphWithOracle<'a, O> {
    /// The underlying graph structure
    graph: &'a ClassicGraph,
    /// Phantom data to carry the oracle type
    _phantom: std::marker::PhantomData<O>,
}

impl<'a, O> ClassicGraphWithOracle<'a, O> {
    /// Creates a new graph wrapper for oracle-based beam search.
    pub fn new(graph: &'a ClassicGraph) -> Self {
        Self {
            graph,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, O: crate::util::Named> crate::util::Named for ClassicGraphWithOracle<'a, O> {
    fn name(&self) -> &str {
        "ClassicGraph(O)"
    }
}

// EVOLVE-BLOCK-START
/// Oracle-agnostic beam search state.
///
/// This generic state works with any oracle type, including:
/// - `RabitqFastScanOracle<B>`: Block-based distance computation (32 vectors at a time)
/// - `RabitqScalarOracle<B>`: Scalar distance computation (one vector at a time)
///
/// The oracle type `O` determines how distances are computed during beam search.
/// Oracle-specific initialization is provided through separate impl blocks.
#[derive(Clone)]
pub struct OracleBeamState<O> {
    /// Set of nodes that have been seen (to avoid re-inserting)
    pub seen: SeenSet,
    /// Beam manager for frontier and visited tracking
    pub beam: Beam,
    /// Set of block indices that have been scanned (used by block-based oracles)
    pub scanned_blocks: SeenSet,
    /// The width of the beam
    pub beam_width: usize,
    /// Optional limit on number of expansions
    pub limit: Option<usize>,
    /// The oracle for distance computation
    pub oracle: O,
    /// Maximum valid vector index (for filtering padding vectors)
    pub max_valid_idx: usize,
    /// Number of distance comparisons performed (used by scalar oracles)
    pub distance_comparisons: usize,
    /// Optional number of candidates to keep for reranking
    pub rerank_n: Option<usize>,
    #[cfg(feature = "record_beams")]
    /// Snapshots of the beam (frontier ids) after each expansion
    pub beam_snapshots: Vec<Vec<IndexT>>,
}

/// Common methods for all oracle types.
impl<O> OracleBeamState<O> {
    /// Returns the number of blocks that have been scanned.
    pub fn num_scanned_blocks(&self) -> usize {
        self.scanned_blocks.len()
    }

    /// Returns the number of distance comparisons performed.
    pub fn num_distance_comparisons(&self) -> usize {
        self.distance_comparisons
    }

    #[cfg(feature = "record_beams")]
    pub fn get_beam_snapshots(&self) -> &Vec<Vec<IndexT>> {
        &self.beam_snapshots
    }
}

/// Generic constructor for any oracle implementing `BeamSearchOracle`.
impl<O: BeamSearchOracle> OracleBeamState<O> {
    /// Create a new search state for any oracle implementing `BeamSearchOracle`.
    ///
    /// # Arguments
    /// * `oracle` - The oracle instance (already initialized with query)
    /// * `start` - Starting node index
    /// * `beam_width` - Width of the beam
    /// * `limit` - Optional limit on number of expansions
    /// * `rerank_n` - Optional number of candidates to track for reranking
    pub fn new(
        oracle: O,
        start: IndexT,
        beam_width: usize,
        limit: Option<usize>,
        rerank_n: Option<usize>,
    ) -> Self {
        let max_valid_idx = oracle.max_valid_idx();
        let num_blocks = oracle.num_blocks();

        // Use bit vectors for fast seen/scanned tracking
        let mut seen = SeenSet::new(max_valid_idx, (beam_width + 10) * SIZE_HINT_FACTOR);
        seen.insert(start);

        let mut scanned_blocks =
            SeenSet::new(num_blocks.max(1), (beam_width + 10) * SIZE_HINT_FACTOR);

        // Initialize using the oracle's initialization method
        let (initial_distance, additional_distances, distance_comparisons) =
            oracle.initialize(start, &mut seen, &mut scanned_blocks);

        let beam_length = rerank_n.map(|n| n.max(beam_width)).unwrap_or(beam_width);
        let mut beam = Beam::new(beam_length, max_valid_idx);
        beam.insert((start, initial_distance));
        beam.batch_insert(&additional_distances);

        OracleBeamState {
            seen,
            beam,
            scanned_blocks,
            beam_width,
            limit,
            oracle,
            max_valid_idx,
            distance_comparisons,
            rerank_n,
            #[cfg(feature = "record_beams")]
            beam_snapshots: Vec::new(),
        }
    }

    pub fn reset_for_query(&mut self, oracle: O, start: IndexT) {
        self.seen.reset();
        self.beam.reset();
        self.scanned_blocks.reset();
        self.distance_comparisons = 0;
        self.oracle = oracle;

        self.seen.insert(start);
        let (initial_distance, additional_distances, distance_comparisons) = self
            .oracle
            .initialize(start, &mut self.seen, &mut self.scanned_blocks);
        self.beam.insert((start, initial_distance));
        self.beam.batch_insert(&additional_distances);
        self.distance_comparisons += distance_comparisons;
    }
}

// ============================================================================
// Generic BeamSearchable implementation for any BeamSearchOracle
// ============================================================================

impl<O: BeamSearchOracle + 'static> BeamSearchable for ClassicGraphWithOracle<'_, O> {
    type Node = IndexT;
    type Neighbor = IndexT;
    type SearchState = OracleBeamState<O>;

    fn expand_node(&self, state: &mut Self::SearchState, node: Self::Node) {
        let current = node;

        // Get neighbors from the graph and process them using the oracle
        let neighbors = self.graph.neighbors(current);

        let mut new_candidates = Vec::new();
        let comparisons = state.oracle.process_neighbors(
            neighbors.iter(),
            &mut state.seen,
            &mut state.scanned_blocks,
            &mut new_candidates,
            None,
        );
        state.beam.batch_insert(&new_candidates);
        state.distance_comparisons += comparisons;
    }
    // EVOLVE-BLOCK-END

    fn next_node(&self, search_state: &mut Self::SearchState) -> Option<Self::Node> {
        if let Some(limit) = search_state.limit {
            if search_state.beam.visited_count() >= limit {
                return None;
            }
        }
        search_state.beam.get_next()
    }

    fn get_results(&self, state: &mut Self::SearchState) -> Vec<IndexT> {
        state
            .beam
            .get_results()
            .into_iter()
            .take(state.rerank_n.unwrap_or(state.beam_width))
            .collect()
    }
}

/// Convenience constructor for RaBitQ FastScan (block-based) oracle.
impl<'a, const B: usize> OracleBeamState<RabitqFastScanOracle<'a, B>> {
    /// Create a new search state for RaBitQ FastScan-based beam search.
    pub fn new_fastscan(
        fastscan: &'a RabitqFastScan<B>,
        query: &[f32],
        start: IndexT,
        beam_width: usize,
        limit: Option<usize>,
        rerank_n: Option<usize>,
    ) -> Self {
        let oracle = fastscan.make_oracle(query);
        OracleBeamState::new(oracle, start, beam_width, limit, rerank_n)
    }
}

/// Generic beam search function that works with any oracle implementing `BeamSearchOracle`.
///
/// This is the most flexible way to perform beam search - you create the oracle yourself
/// and pass it directly. This allows using any custom oracle that implements the trait.
///
/// # Example
/// ```ignore
/// let oracle = fastscan.make_oracle(&query);
/// let (results, stats) = beam_search_with_oracle(&graph, oracle, start, beam_width, limit);
/// ```
///
/// # Returns
/// A tuple of (result_ids, final_state) where the state contains statistics like
/// `num_distance_comparisons()` and `num_scanned_blocks()`.
pub fn beam_search_with_oracle<O: BeamSearchOracle + 'static>(
    graph: &ClassicGraph,
    oracle: O,
    start: IndexT,
    beam_width: usize,
    limit: Option<usize>,
    rerank_n: Option<usize>,
) -> (Vec<IndexT>, OracleBeamState<O>) {
    let wrapper = ClassicGraphWithOracle::<O>::new(graph);
    let mut state = OracleBeamState::new(oracle, start, beam_width, limit, rerank_n);
    wrapper.beam_search(&mut state);
    let results = wrapper.get_results(&mut state);
    (results, state)
}

/// Convenience function to perform beam search on a ClassicGraph using RaBitQ FastScan.
///
/// # Arguments
/// * `graph` - The graph structure
/// * `fastscan` - The RaBitQ FastScan index
/// * `query` - Query vector (dimension must match the FastScan's B parameter)
/// * `start` - Starting node index
/// * `beam_width` - Width of the beam
/// * `limit` - Optional limit on expansions
///
/// # Returns
/// A tuple of (result_ids, num_scanned_blocks)
#[inline(never)]
pub fn beam_search_rabitq<const B: usize>(
    graph: &ClassicGraph,
    fastscan: &RabitqFastScan<B>,
    query: &[f32],
    start: IndexT,
    beam_width: usize,
    limit: Option<usize>,
    rerank_n: Option<usize>,
) -> (Vec<IndexT>, usize) {
    let wrapper = ClassicGraphWithOracle::<RabitqFastScanOracle<'static, B>>::new(graph);

    // Create state with proper lifetime
    let state = OracleBeamState::new_fastscan(fastscan, query, start, beam_width, limit, rerank_n);

    // SAFETY: We're transmuting the lifetime here because the BeamSearchable trait
    // requires SearchState to be 'static, but our state borrows from fastscan.
    // This is safe because we consume the state before returning, ensuring
    // the fastscan reference remains valid throughout.
    let state: OracleBeamState<RabitqFastScanOracle<'static, B>> =
        unsafe { std::mem::transmute(state) };

    let mut state = state;
    wrapper.beam_search(&mut state);
    let results = wrapper.get_results(&mut state);
    let num_scanned = state.num_scanned_blocks();

    (results, num_scanned)
}

pub fn beam_search_rabitq_with_state<const B: usize>(
    graph: &ClassicGraph,
    fastscan: &RabitqFastScan<B>,
    query: &[f32],
    start: IndexT,
    state: &mut OracleBeamState<RabitqFastScanOracle<'static, B>>,
) -> (Vec<IndexT>, usize) {
    let wrapper = ClassicGraphWithOracle::<RabitqFastScanOracle<'static, B>>::new(graph);

    // SAFETY: The oracle borrows from fastscan/query for this call only.
    let oracle = RabitqFastScanOracle::new(fastscan, query);
    let oracle: RabitqFastScanOracle<'static, B> = unsafe { std::mem::transmute(oracle) };

    state.reset_for_query(oracle, start);
    wrapper.beam_search(state);
    let results = wrapper.get_results(state);
    let num_scanned = state.num_scanned_blocks();

    (results, num_scanned)
}

/// Convenience constructor for RaBitQ Scalar (one-at-a-time) oracle.
impl<'a, const B: usize> OracleBeamState<RabitqScalarOracle<'a, B>> {
    /// Create a new search state for RaBitQ scalar beam search.
    pub fn new_scalar(
        fastscan: &'a RabitqFastScan<B>,
        query: &[f32],
        start: IndexT,
        beam_width: usize,
        limit: Option<usize>,
        rerank_n: Option<usize>,
    ) -> Self {
        let oracle = RabitqScalarOracle::new(fastscan, query);
        OracleBeamState::new(oracle, start, beam_width, limit, rerank_n)
    }
}

/// Convenience function to perform beam search on a ClassicGraph using RaBitQ scalar distances.
///
/// Unlike `beam_search_rabitq` which uses block-based FastScan (32 vectors at a time),
/// this function computes distances one at a time using the scalar implementation.
///
/// # Arguments
/// * `graph` - The graph structure
/// * `fastscan` - The RaBitQ FastScan index (used for binary codes and factors)
/// * `query` - Query vector (dimension must match the FastScan's B parameter)
/// * `start` - Starting node index
/// * `beam_width` - Width of the beam
/// * `limit` - Optional limit on expansions
///
/// # Returns
/// A tuple of (result_ids, num_distance_comparisons)
#[inline(never)]
pub fn beam_search_rabitq_scalar<const B: usize>(
    graph: &ClassicGraph,
    fastscan: &RabitqFastScan<B>,
    query: &[f32],
    start: IndexT,
    beam_width: usize,
    limit: Option<usize>,
    rerank_n: Option<usize>,
) -> (Vec<IndexT>, usize) {
    let wrapper = ClassicGraphWithOracle::<RabitqScalarOracle<'static, B>>::new(graph);

    // Create state with proper lifetime
    let state = OracleBeamState::new_scalar(fastscan, query, start, beam_width, limit, rerank_n);

    // SAFETY: Same as beam_search_rabitq - transmuting lifetime is safe because
    // we consume the state before returning.
    let state: OracleBeamState<RabitqScalarOracle<'static, B>> =
        unsafe { std::mem::transmute(state) };

    let mut state = state;
    wrapper.beam_search(&mut state);
    let results = wrapper.get_results(&mut state);
    let num_comparisons = state.num_distance_comparisons();

    (results, num_comparisons)
}

pub fn beam_search_rabitq_scalar_with_state<const B: usize>(
    graph: &ClassicGraph,
    fastscan: &RabitqFastScan<B>,
    query: &[f32],
    start: IndexT,
    state: &mut OracleBeamState<RabitqScalarOracle<'static, B>>,
) -> (Vec<IndexT>, usize) {
    let wrapper = ClassicGraphWithOracle::<RabitqScalarOracle<'static, B>>::new(graph);

    // SAFETY: The oracle borrows from fastscan/query for this call only.
    let oracle = RabitqScalarOracle::new(fastscan, query);
    let oracle: RabitqScalarOracle<'static, B> = unsafe { std::mem::transmute(oracle) };

    state.reset_for_query(oracle, start);
    wrapper.beam_search(state);
    let results = wrapper.get_results(state);
    let num_comparisons = state.num_distance_comparisons();

    (results, num_comparisons)
}

#[cfg(test)]
mod rabitq_beam_search_tests {
    use super::*;
    use crate::data_handling::dataset::VectorDataset;

    const TEST_DIM: usize = 128;

    /// Creates a random dataset with pseudo-random values.
    fn make_random_dataset(n: usize, dim: usize, seed: u64) -> VectorDataset<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for d in 0..dim {
                let mut hasher = DefaultHasher::new();
                (seed, i, d).hash(&mut hasher);
                let hash = hasher.finish();
                data[i * dim + d] = ((hash as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
            }
        }
        VectorDataset::new(data.into_boxed_slice(), n, dim)
    }

    /// Creates a simple graph where each node connects to a few nearby nodes.
    fn make_simple_graph(n: usize, max_degree: usize) -> ClassicGraph {
        let mut graph = ClassicGraph::new(n as IndexT, max_degree);
        for i in 0..n {
            for j in 1..=max_degree.min(n - 1) {
                let neighbor = (i + j) % n;
                if neighbor != i && graph.degree(i as IndexT) < max_degree {
                    graph.add_edge(i as IndexT, neighbor as IndexT);
                }
            }
        }
        graph
    }

    #[test]
    fn test_beam_search_rabitq_basic() {
        // Create a dataset with vectors aligned to 32-block boundary
        let n = 64; // 2 blocks
        let dataset = make_random_dataset(n, TEST_DIM, 42);
        let fastscan = RabitqFastScan::<TEST_DIM>::from_f32_dataset(&dataset);

        // Create a simple graph
        let graph = make_simple_graph(n, 4);

        // Create a query
        let query: Vec<f32> = (0..TEST_DIM)
            .map(|d| if d % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        // Run beam search
        let (results, num_scanned) = beam_search_rabitq(
            &graph, &fastscan, &query, 0,  // start node
            10, // beam width
            None, None,
        );

        // Basic sanity checks
        assert!(!results.is_empty(), "Results should not be empty");
        assert!(results.len() <= 10, "Should not exceed beam width");
        assert!(num_scanned > 0, "Should have scanned at least one block");

        // Check that results are valid indices
        for &idx in &results {
            assert!(
                (idx as usize) < n,
                "Result index {} should be less than n={}",
                idx,
                n
            );
        }
    }

    #[test]
    fn test_beam_search_rabitq_with_limit() {
        let n = 64;
        let dataset = make_random_dataset(n, TEST_DIM, 123);
        let fastscan = RabitqFastScan::<TEST_DIM>::from_f32_dataset(&dataset);
        let graph = make_simple_graph(n, 4);

        let query: Vec<f32> = (0..TEST_DIM).map(|d| (d as f32) * 0.01).collect();

        // Run with expansion limit
        let (results, num_scanned_limited) = beam_search_rabitq(
            &graph,
            &fastscan,
            &query,
            0,
            10,
            Some(5), // limit to 5 expansions
            None,
        );

        // Run without limit for comparison
        let (_results_full, num_scanned_full) =
            beam_search_rabitq(&graph, &fastscan, &query, 0, 10, None, None);

        assert!(!results.is_empty());
        // With a limit, we should scan fewer or equal blocks
        assert!(
            num_scanned_limited <= num_scanned_full,
            "Limited search should scan fewer blocks"
        );
    }

    #[test]
    fn test_beam_search_rabitq_deterministic() {
        let n = 64;
        let dataset = make_random_dataset(n, TEST_DIM, 456);
        let fastscan = RabitqFastScan::<TEST_DIM>::from_f32_dataset(&dataset);
        let graph = make_simple_graph(n, 4);

        let query: Vec<f32> = (0..TEST_DIM).map(|d| (d as f32) / 100.0).collect();

        // Run twice with same parameters
        let (results1, blocks1) = beam_search_rabitq(&graph, &fastscan, &query, 0, 10, None, None);
        let (results2, blocks2) = beam_search_rabitq(&graph, &fastscan, &query, 0, 10, None, None);

        // Results should be identical
        assert_eq!(results1, results2, "Results should be deterministic");
        assert_eq!(blocks1, blocks2, "Block counts should be deterministic");
    }

    #[test]
    fn test_beam_search_rabitq_block_efficiency() {
        // Test that we're efficiently using block-wise distance computation
        let n = 128; // 4 blocks
        let dataset = make_random_dataset(n, TEST_DIM, 789);
        let fastscan = RabitqFastScan::<TEST_DIM>::from_f32_dataset(&dataset);

        // Create a graph where all neighbors are in the same block
        let mut graph = ClassicGraph::new(n as IndexT, 8);
        // Node 0 connects to nodes 1-8 (all in block 0)
        for i in 1..=8 {
            graph.add_edge(0, i as IndexT);
        }
        // Node 1 connects to nodes 2-9 (mostly in block 0)
        for i in 2..=9 {
            if i < n {
                graph.add_edge(1, i as IndexT);
            }
        }

        let query: Vec<f32> = vec![1.0; TEST_DIM];

        let (results, num_scanned) =
            beam_search_rabitq(&graph, &fastscan, &query, 0, 10, Some(3), None);

        // With neighbors in the same block, we should scan very few blocks
        // Starting block (0) + possibly a few more
        assert!(
            num_scanned <= 2,
            "Should scan at most 2 blocks when neighbors are clustered, got {}",
            num_scanned
        );
        assert!(!results.is_empty());
    }

    #[test]
    fn test_beam_search_rabitq_adds_all_block_distances() {
        // Verify that when a block is scanned, all 32 vectors are considered
        let n = 32; // Exactly 1 block
        let dataset = make_random_dataset(n, TEST_DIM, 111);
        let fastscan = RabitqFastScan::<TEST_DIM>::from_f32_dataset(&dataset);

        // Minimal graph: node 0 connects to node 1
        let mut graph = ClassicGraph::new(n as IndexT, 2);
        graph.add_edge(0, 1);
        graph.add_edge(1, 0);

        let query: Vec<f32> = vec![0.5; TEST_DIM];

        // With beam_width = 32, we should get all nodes since they're all in one block
        let (results, num_scanned) =
            beam_search_rabitq(&graph, &fastscan, &query, 0, 32, None, None);

        assert_eq!(num_scanned, 1, "Should only scan 1 block");
        assert_eq!(
            results.len(),
            32,
            "Should return all 32 nodes from the single block"
        );
    }

    #[test]
    fn test_beam_search_rabitq_different_start_nodes() {
        let n = 64;
        let dataset = make_random_dataset(n, TEST_DIM, 222);
        let fastscan = RabitqFastScan::<TEST_DIM>::from_f32_dataset(&dataset);
        let graph = make_simple_graph(n, 4);

        let query: Vec<f32> = vec![1.0; TEST_DIM];

        // Search from different starting nodes
        let (results_0, _) = beam_search_rabitq(&graph, &fastscan, &query, 0, 10, None, None);
        let (results_32, _) = beam_search_rabitq(&graph, &fastscan, &query, 32, 10, None, None);

        // Both should return valid results
        assert!(!results_0.is_empty());
        assert!(!results_32.is_empty());

        // Since they're searching the same dataset with same query,
        // they should converge to similar (though not necessarily identical) results
        // with enough iterations
    }

    #[test]
    fn test_scalar_oracle_matches_fastscan() {
        // Test that the scalar oracle produces similar distances to fastscan
        const DIM: usize = 128;
        let n = 64; // 2 blocks
        let dataset = make_random_dataset(n, DIM, 42);
        let fastscan = RabitqFastScan::<DIM>::from_f32_dataset(&dataset);

        // Create a query
        let query: Vec<f32> = (0..DIM).map(|d| (d as f64 * 0.1).sin() as f32).collect();

        // Create both oracles
        let fastscan_oracle = fastscan.make_oracle(&query);
        let scalar_oracle = RabitqScalarOracle::<DIM>::new(&fastscan, &query);

        // Compare distances for all vectors
        for block_idx in 0..fastscan.num_blocks() {
            let fastscan_distances = fastscan_oracle.compute_block_distances(block_idx);

            for &(idx, fastscan_dist) in &fastscan_distances {
                if idx >= n {
                    continue; // Skip padding
                }
                let scalar_dist = scalar_oracle.compute_distance(idx);

                // Allow some tolerance due to quantization differences
                let diff = (fastscan_dist - scalar_dist).abs();
                let relative_diff = if fastscan_dist.abs() > 1e-6 {
                    diff / fastscan_dist.abs()
                } else {
                    diff
                };

                assert!(
                    relative_diff < 0.01 || diff < 0.1,
                    "Distance mismatch at idx {}: fastscan={}, scalar={}, diff={}",
                    idx,
                    fastscan_dist,
                    scalar_dist,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_scalar_beam_search_basic() {
        const DIM: usize = 128;
        let n = 64;
        let dataset = make_random_dataset(n, DIM, 123);
        let fastscan = RabitqFastScan::<DIM>::from_f32_dataset(&dataset);
        let graph = make_simple_graph(n, 4);

        let query: Vec<f32> = (0..DIM).map(|d| (d as f32) * 0.01).collect();

        // Run both scalar and fastscan beam search
        let (scalar_results, _) =
            beam_search_rabitq_scalar(&graph, &fastscan, &query, 0, 10, None, None);
        let (fastscan_results, _) =
            beam_search_rabitq(&graph, &fastscan, &query, 0, 10, None, None);

        // Both should return results
        assert!(
            !scalar_results.is_empty(),
            "Scalar search returned no results"
        );
        assert!(
            !fastscan_results.is_empty(),
            "FastScan search returned no results"
        );

        // Results should be similar (not necessarily identical due to tie-breaking)
        // Check that there's significant overlap in top results
        let scalar_set: std::collections::HashSet<_> = scalar_results.iter().take(5).collect();
        let fastscan_set: std::collections::HashSet<_> = fastscan_results.iter().take(5).collect();
        let overlap = scalar_set.intersection(&fastscan_set).count();

        // Expect at least some overlap in top-5 results
        assert!(
            overlap >= 2,
            "Too little overlap between scalar and fastscan results: {} out of 5",
            overlap
        );
    }
}
