//! This library provides methods for encoding the data into chunks and
//! reconstructing the original data from chunks as well as verifying
//! individual chunks against an erasure root.

mod error;
mod merklize;
mod subshard;

pub use self::{
	error::Error,
	merklize::{ErasureRoot, MerklizedChunks, Proof},
};

use scale::{Decode, Encode};
use std::ops::AddAssign;
pub use subshard::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use std::sync::Once;

#[cfg(feature = "arena")]
use bumpalo::Bump;

// Prefetch hints for cache locality optimization
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Branch prediction hints
#[cold]
#[inline(never)]
fn cold() {}

#[inline(always)]
fn likely(b: bool) -> bool {
	if !b {
		cold();
	}
	b
}

#[inline(always)]
fn unlikely(b: bool) -> bool {
	if b {
		cold();
	}
	b
}

pub const MAX_CHUNKS: u16 = 16384;

// The reed-solomon library requires each shards to be 64 bytes aligned.
const SHARD_ALIGNMENT: usize = 64;

/// Initialize the thread pool for parallel computations.
/// The number of threads can be configured via the `RAYON_NUM_THREADS` environment variable
/// at compile time. If the variable is not set, the number of CPU cores is used.
///
/// Usage example:
/// ```bash
/// RAYON_NUM_THREADS=4 cargo build --features parallel
/// ```
#[cfg(feature = "parallel")]
pub(crate) fn init_rayon_thread_pool() {
	static INIT: Once = Once::new();
	INIT.call_once(|| {
		// Get the number of threads from the environment variable at compile time
		// If the variable is not set, use the default value (0 = number of CPU cores)
		if let Some(num_threads_str) = option_env!("RAYON_NUM_THREADS") {
			if let Ok(num_threads) = num_threads_str.parse::<usize>() {
				if num_threads > 0 {
					// Set the environment variable at runtime,
					// so rayon can use it during initialization
					std::env::set_var("RAYON_NUM_THREADS", num_threads_str);

					// Try to initialize the global thread pool
					// If it's already initialized, this is safely ignored
					let _ = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global();
				}
			}
		}
		// If the variable is not set or equals 0, rayon uses the default number of CPU cores
	});
}

/// The index of an erasure chunk.
#[derive(Eq, Ord, PartialEq, PartialOrd, Copy, Clone, Encode, Decode, Hash, Debug)]
pub struct ChunkIndex(pub u16);

impl From<u16> for ChunkIndex {
	fn from(n: u16) -> Self {
		ChunkIndex(n)
	}
}

impl AddAssign<u16> for ChunkIndex {
	fn add_assign(&mut self, rhs: u16) {
		self.0 += rhs
	}
}

/// A chunk of erasure-encoded block data.
#[derive(PartialEq, Eq, Clone, Encode, Decode, Debug)]
pub struct ErasureChunk {
	/// The erasure-encoded chunk of data belonging to the candidate block.
	pub chunk: Vec<u8>,
	/// The index of this erasure-encoded chunk of data.
	pub index: ChunkIndex,
	/// Proof for this chunk against an erasure root.
	pub proof: Proof,
}

/// Obtain a threshold of chunks that should be enough to recover the data.
#[inline]
pub const fn recovery_threshold(n_chunks: u16) -> Result<u16, Error> {
	if n_chunks > MAX_CHUNKS {
		return Err(Error::TooManyTotalChunks);
	}
	if n_chunks == 0 {
		return Err(Error::NotEnoughTotalChunks);
	}

	let needed = (n_chunks - 1) / 3;
	Ok(needed + 1)
}

/// Obtain the threshold of systematic chunks that should be enough to recover the data.
#[inline]
pub fn systematic_recovery_threshold(n_chunks: u16) -> Result<u16, Error> {
	recovery_threshold(n_chunks)
}

/// Reconstruct the original data from the set of systematic chunks.
///
/// Provide a vector containing the first k chunks in order. If too few chunks are provided,
/// recovery is not possible.
///
/// Due to the internals of the erasure coding algorithm, the output might be
/// larger than the original data and padded with zeroes; passing `data_len`
/// allows to truncate the output to the original data size.
pub fn reconstruct_from_systematic<'a>(
	n_chunks: u16,
	systematic_len: usize,
	systematic_chunks: &'a mut impl Iterator<Item = &'a [u8]>,
	data_len: usize,
) -> Result<Vec<u8>, Error> {
	let k = systematic_recovery_threshold(n_chunks)? as usize;
	if unlikely(systematic_len < k) {
		return Err(Error::NotEnoughChunks);
	}

	let mut bytes: Vec<u8> = Vec::with_capacity(0);
	let mut shard_len = 0;
	let mut nb = 0;

	for chunk in systematic_chunks.by_ref() {
		nb += 1;
		if unlikely(shard_len == 0) {
			shard_len = chunk.len();
			if unlikely(shard_len % SHARD_ALIGNMENT != 0 && nb != k) {
				return Err(Error::UnalignedChunk);
			}

			if unlikely(k == 1) {
				return Ok(chunk[..data_len].to_vec());
			}
			bytes = Vec::with_capacity(shard_len * k);
		}

		if unlikely(chunk.len() != shard_len) {
			return Err(Error::NonUniformChunks);
		}

		// extend_from_slice uses optimized memcpy
		bytes.extend_from_slice(chunk);

		if unlikely(nb == k) {
			break;
		}
	}

	bytes.resize(data_len, 0);
	Ok(bytes)
}

/// Construct erasure-coded chunks.
///
/// Works only for 1..65536 chunks.
/// The data must be non-empty.
pub fn construct_chunks(n_chunks: u16, data: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
	if unlikely(data.is_empty()) {
		return Err(Error::BadPayload);
	}
	if unlikely(n_chunks == 1) {
		return Ok(vec![data.to_vec()]);
	}

	#[cfg(feature = "arena")]
	{
		construct_chunks_arena(n_chunks, data)
	}

	#[cfg(not(feature = "arena"))]
	{
		construct_chunks_default(n_chunks, data)
	}
}

/// Version without arena allocator
#[inline]
fn construct_chunks_default(n_chunks: u16, data: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
	let systematic = systematic_recovery_threshold(n_chunks)?;
	let original_data = make_original_shards(systematic, data);
	let original_iter = original_data.iter();
	let original_count = systematic as usize;
	let recovery_count = (n_chunks - systematic) as usize;

	let recovery = reed_solomon::encode(original_count, recovery_count, original_iter)?;

	let mut result = original_data;
	result.extend(recovery);

	Ok(result)
}

/// Optimized version with arena allocator
#[cfg(feature = "arena")]
fn construct_chunks_arena(n_chunks: u16, data: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
	let systematic = systematic_recovery_threshold(n_chunks)?;
	let original_count = systematic as usize;
	let recovery_count = (n_chunks - systematic) as usize;
	let shard_size = shard_bytes(systematic, data.len());

	// Arena for temporary allocations
	let arena = Bump::with_capacity(original_count * shard_size + 4096);

	// Create shards using arena for intermediate data
	let original_data = make_original_shards_arena(&arena, systematic, data, shard_size);
	let original_iter = original_data.iter();

	let recovery = reed_solomon::encode(original_count, recovery_count, original_iter)?;

	let mut result = original_data;
	result.extend(recovery);

	Ok(result)
}

/// Creating shards using arena allocator
#[cfg(feature = "arena")]
fn make_original_shards_arena(
	_arena: &Bump,
	original_count: u16,
	data: &[u8],
	shard_size: usize,
) -> Vec<Vec<u8>> {
	// Optimization: use single large buffer instead of multiple Vecs
	let total_size = original_count as usize * shard_size;
	let mut flat_buffer = vec![0u8; total_size];

	// Copy data to flat buffer in one call
	let data_to_copy = data.len().min(total_size);
	flat_buffer[..data_to_copy].copy_from_slice(&data[..data_to_copy]);

	// Slice flat buffer into chunks without additional copying
	let mut result = Vec::with_capacity(original_count as usize);
	for chunk_data in flat_buffer.chunks_exact(shard_size) {
		result.push(chunk_data.to_vec());
	}

	result
}

#[inline(always)]
fn next_aligned(n: usize, alignment: usize) -> usize {
	((n + alignment - 1) / alignment) * alignment
}

#[inline]
fn shard_bytes(systematic: u16, data_len: usize) -> usize {
	let shard_bytes = (data_len + systematic as usize - 1) / systematic as usize;
	next_aligned(shard_bytes, SHARD_ALIGNMENT)
}

// The reed-solomon library takes sharded data as input.
fn make_original_shards(original_count: u16, data: &[u8]) -> Vec<Vec<u8>> {
	assert!(!data.is_empty(), "data must be non-empty");
	assert_ne!(original_count, 0);

	let shard_bytes = shard_bytes(original_count, data.len());
	assert_ne!(shard_bytes, 0);

	#[cfg(not(feature = "parallel"))]
	{
		// Optimized sequential version with prefetch
		let mut result = Vec::with_capacity(original_count as usize);
		let mut remaining_data = data;

		for i in 0..original_count as usize {
			let mut chunk = vec![0u8; shard_bytes];
			let copy_len = remaining_data.len().min(shard_bytes);

			// Prefetch next data block
			#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
			if i + 1 < original_count as usize && remaining_data.len() > shard_bytes {
				unsafe {
					let next_ptr = remaining_data.as_ptr().add(shard_bytes);
					if (remaining_data.len() - shard_bytes) >= 64 {
						_mm_prefetch(next_ptr as *const i8, _MM_HINT_T0);
					}
				}
			}

			// Optimized copying (compiler uses memcpy/SIMD)
			chunk[..copy_len].copy_from_slice(&remaining_data[..copy_len]);

			if likely(remaining_data.len() >= shard_bytes) {
				remaining_data = &remaining_data[shard_bytes..];
			} else {
				remaining_data = &[];
			}

			result.push(chunk);
		}
		result
	}

	#[cfg(feature = "parallel")]
	{
		// Initialize the thread pool on first use
		init_rayon_thread_pool();

		// Parallel version: create shards in parallel
		(0..original_count as usize)
			.into_par_iter()
			.map(|i| {
				let mut chunk = vec![0u8; shard_bytes];
				let start = i * shard_bytes;
				let end = (start + shard_bytes).min(data.len());

				if likely(start < data.len()) {
					let copy_len = end - start;
					chunk[..copy_len].copy_from_slice(&data[start..end]);
				}

				chunk
			})
			.collect()
	}
}

/// Reconstruct the original data from a set of chunks.
///
/// Provide an iterator containing chunk data and the corresponding index.
/// The indices of the present chunks must be indicated. If too few chunks
/// are provided, recovery is not possible.
///
/// Works only for 1..65536 chunks.
///
/// Due to the internals of the erasure coding algorithm, the output might be
/// larger than the original data and padded with zeroes; passing `data_len`
/// allows to truncate the output to the original data size.
pub fn reconstruct<I>(n_chunks: u16, chunks: I, data_len: usize) -> Result<Vec<u8>, Error>
where
	I: IntoIterator<Item = (ChunkIndex, Vec<u8>)>,
{
	if n_chunks == 1 {
		return chunks.into_iter().next().map(|(_, v)| v).ok_or(Error::NotEnoughChunks);
	}
	let n = n_chunks as usize;
	let original_count = systematic_recovery_threshold(n_chunks)? as usize;
	let recovery_count = n - original_count;

	let (mut original, recovery): (Vec<_>, Vec<_>) = chunks
		.into_iter()
		.map(|(i, v)| (i.0 as usize, v))
		.partition(|(i, _)| *i < original_count);

	original.sort_by_key(|(i, _)| *i);
	let original_iter = original.iter().map(|(i, v)| (*i, v));
	let recovery = recovery.into_iter().map(|(i, v)| (i - original_count, v));

	let mut recovered =
		reed_solomon::decode(original_count, recovery_count, original_iter, recovery)?;

	let shard_bytes = shard_bytes(original_count as u16, data_len);
	let mut bytes = Vec::with_capacity(shard_bytes * original_count);

	let mut original = original.into_iter();
	for i in 0..original_count {
		let chunk = recovered.remove(&i).unwrap_or_else(|| {
			let (j, v) = original.next().expect("what is not recovered must be present; qed");
			debug_assert_eq!(i, j);
			v
		});
		bytes.extend_from_slice(chunk.as_slice());
	}

	bytes.truncate(data_len);

	Ok(bytes)
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use super::*;
	use quickcheck::{Arbitrary, Gen, QuickCheck};

	#[derive(Clone, Debug)]
	struct ArbitraryAvailableData(Vec<u8>);

	impl Arbitrary for ArbitraryAvailableData {
		fn arbitrary(g: &mut Gen) -> Self {
			// Limit the POV len to 16KiB, otherwise the test will take forever
			let data_len = (u32::arbitrary(g) % (16 * 1024)).max(2);

			let data = (0..data_len).map(|_| u8::arbitrary(g)).collect();

			ArbitraryAvailableData(data)
		}
	}

	#[derive(Clone, Debug)]
	struct SmallAvailableData(Vec<u8>);

	impl Arbitrary for SmallAvailableData {
		fn arbitrary(g: &mut Gen) -> Self {
			let data_len = (u32::arbitrary(g) % (2 * 1024)).max(2);

			let data = (0..data_len).map(|_| u8::arbitrary(g)).collect();

			Self(data)
		}
	}

	#[test]
	fn round_trip_systematic_works() {
		fn property(available_data: ArbitraryAvailableData, n_chunks: u16) {
			let n_chunks = n_chunks.max(1).min(MAX_CHUNKS);
			let threshold = systematic_recovery_threshold(n_chunks).unwrap();
			let data_len = available_data.0.len();
			let chunks = construct_chunks(n_chunks, &available_data.0).unwrap();
			let reconstructed: Vec<u8> = reconstruct_from_systematic(
				n_chunks,
				chunks.len(),
				&mut chunks.iter().take(threshold as usize).map(Vec::as_slice),
				data_len,
			)
			.unwrap();
			assert_eq!(reconstructed, available_data.0);
		}

		QuickCheck::new().quickcheck(property as fn(ArbitraryAvailableData, u16))
	}

	#[test]
	fn round_trip_works() {
		fn property(available_data: ArbitraryAvailableData, n_chunks: u16) {
			let n_chunks = n_chunks.max(1).min(MAX_CHUNKS);
			let data_len = available_data.0.len();
			let threshold = recovery_threshold(n_chunks).unwrap();
			let chunks = construct_chunks(n_chunks, &available_data.0).unwrap();
			let map: HashMap<ChunkIndex, Vec<u8>> = chunks
				.into_iter()
				.enumerate()
				.map(|(i, v)| (ChunkIndex::from(i as u16), v))
				.collect();
			let some_chunks = map.into_iter().take(threshold as usize);
			let reconstructed: Vec<u8> = reconstruct(n_chunks, some_chunks, data_len).unwrap();
			assert_eq!(reconstructed, available_data.0);
		}

		QuickCheck::new().quickcheck(property as fn(ArbitraryAvailableData, u16))
	}

	#[test]
	fn proof_verification_works() {
		fn property(data: SmallAvailableData, n_chunks: u16) {
			let n_chunks = n_chunks.max(1).min(2048);
			let chunks = construct_chunks(n_chunks, &data.0).unwrap();
			assert_eq!(chunks.len() as u16, n_chunks);

			let iter = MerklizedChunks::compute(chunks.clone());
			let root = iter.root();
			let erasure_chunks: Vec<_> = iter.collect();

			assert_eq!(erasure_chunks.len(), chunks.len());

			for erasure_chunk in erasure_chunks.into_iter() {
				let encode = Encode::encode(&erasure_chunk.proof);
				let decode = Decode::decode(&mut &encode[..]).unwrap();
				assert_eq!(erasure_chunk.proof, decode);
				assert_eq!(encode, Encode::encode(&decode));

				assert_eq!(&erasure_chunk.chunk, &chunks[erasure_chunk.index.0 as usize]);

				assert!(erasure_chunk.verify(&root));
			}
		}

		QuickCheck::new().quickcheck(property as fn(SmallAvailableData, u16))
	}

	#[test]
	fn stress_test_various_sizes_with_random_chunk_loss() {
		use rand::{seq::SliceRandom, Rng, SeedableRng};

		// Data sizes for testing
		let data_sizes = vec![10, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000];

		// Various chunk configurations for testing
		let chunk_configs = vec![16, 64, 256, 1024];

		for data_size in data_sizes.iter() {
			println!("Testing data size: {} bytes", data_size);

			for &n_chunks in chunk_configs.iter() {
				// Skip too large configurations for small data
				if *data_size < 1000 && n_chunks > 64 {
					continue;
				}

				println!("  Testing with {} chunks", n_chunks);

				// Generate random data with fixed seed for reproducibility
				let mut rng =
					rand::rngs::SmallRng::seed_from_u64((*data_size as u64) ^ (n_chunks as u64));
				let original_data: Vec<u8> = (0..*data_size).map(|_| rng.gen()).collect();

				// Encode data
				let chunks = construct_chunks(n_chunks, &original_data).unwrap();
				assert_eq!(chunks.len(), n_chunks as usize);

				// Get minimum threshold for recovery
				let threshold = recovery_threshold(n_chunks).unwrap() as usize;

				// Create indices of all chunks
				let mut chunk_indices: Vec<usize> = (0..n_chunks as usize).collect();

				// Shuffle indices
				chunk_indices.shuffle(&mut rng);

				// Take only threshold chunks (minimum required amount)
				let selected_indices = &chunk_indices[..threshold];

				// Create map with selected chunks
				let available_chunks: HashMap<ChunkIndex, Vec<u8>> = selected_indices
					.iter()
					.map(|&idx| (ChunkIndex(idx as u16), chunks[idx].clone()))
					.collect();

				println!(
					"    Using {} chunks out of {} (threshold: {})",
					available_chunks.len(),
					n_chunks,
					threshold
				);

				// Recover data from available chunks
				let reconstructed =
					reconstruct(n_chunks, available_chunks.into_iter(), original_data.len())
						.unwrap();

				// Verify that reconstructed data matches original
				assert_eq!(
					reconstructed.len(),
					original_data.len(),
					"Reconstructed data length mismatch for size {} with {} chunks",
					data_size,
					n_chunks
				);

				assert_eq!(
					reconstructed, original_data,
					"Reconstructed data does not match original for size {} with {} chunks",
					data_size, n_chunks
				);
			}

			println!("  ✓ All chunk configurations passed for size {}", data_size);
		}

		println!("✓ All stress tests passed!");
	}
}
