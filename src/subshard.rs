//! Manage subshards of 12 bytes of 4ko segments.
//! Subshards are processed in to shards of 64 bytes
//! to benefit from the simd optimisation in the
//! best case.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// Fix segment size.
const SEGMENT_SIZE: usize = 4096;

const SUBSHARD_PER_SEGMENT: usize = ((SEGMENT_SIZE - 1) / SUBSHARD_SIZE) + 1;

/// Segment size with added padding to allow being
/// erasure coded in batch while staying on same points indexes.
const SEGMENT_SIZE_ALIGNED: usize = SUBSHARD_PER_SEGMENT * SUBSHARD_SIZE; // 4104 byte

/// Fix number of shards and subshards.
const N_SHARDS: usize = 342;

/// The number of time the erasure coded shards we want.
const N_REDUNDANCY: usize = 2;

/// The total number of shards, both original and ec one.
const TOTAL_SHARDS: usize = (1 + N_REDUNDANCY) * N_SHARDS;

/// The reed-solomon library requires each shards to be 64 bytes aligned.
const SHARD_MIN_SIZE: usize = SHARD_ALIGNMENT;

/// In shards lower and higher byte of each point are spaced for simd.
const POINT_BYTE_SPACING: usize = SHARD_ALIGNMENT / 2;

/// Number points in each subshards.
const SUBSHARD_POINTS: usize = 6;

/// Size of a point in bytes.
const POINT_SIZE: usize = 2; // gf16

/// Size of a subshard in bytes.
const SUBSHARD_SIZE: usize = POINT_SIZE * SUBSHARD_POINTS; // 12bytes

/// Aligned number of full shard to process subshard.
const SUBSHARD_BATCH_MUL: usize = 3; // 3 * 12 is aligned with 64

/// Number of segments in a aligned batch.
const SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL: usize =
	SUBSHARD_BATCH_MUL * SHARD_MIN_SIZE / SUBSHARD_SIZE; // 16

/// Fix size segment of a larger data.
/// Data is padded when unaligned with
/// the segment size.
#[derive(PartialEq, Eq, Clone, Encode, Decode, Debug)]
pub struct Segment {
	/// Fix size chunk of data.
	pub data: Box<[u8; SEGMENT_SIZE]>,
	/// The index of this segment against its full data.
	pub index: u32,
}

/// Subshard (points in sequential orders).
pub type SubShard = [u8; SUBSHARD_SIZE];

/// Subshard uses some temp memory, so these should be used multiple time instead of allocating.
pub struct SubShardEncoder {
	encoder: reed_solomon::ReedSolomonEncoder,
}

impl SubShardEncoder {
	pub fn new() -> Result<Self, Error> {
		Ok(Self {
			encoder: reed_solomon::ReedSolomonEncoder::new(
				N_SHARDS,
				N_REDUNDANCY * N_SHARDS,
				SHARD_MIN_SIZE * SUBSHARD_BATCH_MUL,
			)?,
		})
	}

	/// Construct erasure-coded chunks.
	/// Segement input must be ordered by index and consecutive.
	/// Data must be less than MAX_SUB_OPTIMAL_SIZE_DATA.
	pub fn construct_chunks(
		&mut self,
		data: &[Segment],
	) -> Result<Vec<Box<[SubShard; TOTAL_SHARDS]>>, Error> {
		if data.len() > SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL {
			return Err(Error::BadPayload);
		}
		let mut next = 0;
		for s in data.iter() {
			if s.index != next {
				return Err(Error::BadPayload);
			}
			next += 1;
		}
		let mut result = vec![
			Box::new([[0u8; SUBSHARD_SIZE]; TOTAL_SHARDS]);
			SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL
		];

		let mut shard = [0u8; SUBSHARD_BATCH_MUL * SHARD_MIN_SIZE];
		for shard_a in 0..N_SHARDS {
			let mut shard_i = 0;
			for segment_i in 0..data.len() {
				for point_i in 0..SUBSHARD_POINTS {
					let data_i = (point_i * N_SHARDS) * 2 + shard_a * 2;
					let point = if data_i < SEGMENT_SIZE {
						(data[segment_i].data[data_i], data[segment_i].data[data_i + 1])
					} else {
						(0, 0)
					};
					shard[shard_i] = point.0;
					shard[shard_i + POINT_BYTE_SPACING] = point.1;
					result[segment_i][shard_a][point_i * 2] = point.0;
					result[segment_i][shard_a][point_i * 2 + 1] = point.1;
					shard_i += 1;
					if shard_i % POINT_BYTE_SPACING == 0 {
						shard_i += POINT_BYTE_SPACING;
					}
				}
			}
			self.encoder.add_original_shard(&shard)?;
		}

		let enco_res = self.encoder.encode()?;
		for (shard_a, data) in enco_res.recovery_iter().enumerate() {
			let mut segment_i = 0;
			let mut data_i = 0;
			while data_i != data.len() {
				for point_i in 0..SUBSHARD_POINTS {
					let point = (data[data_i], data[data_i + 32]);
					data_i += 1;
					if data_i % POINT_BYTE_SPACING == 0 {
						data_i += POINT_BYTE_SPACING;
					}
					result[segment_i][shard_a + N_SHARDS][point_i * 2] = point.0;
					result[segment_i][shard_a + N_SHARDS][point_i * 2 + 1] = point.1;
				}
				segment_i += 1;
			}
		}
		return Ok(result);
	}
}

/// Subshard uses some temp memory, so these should be used multiple time instead of allocating.
pub struct SubShardDecoder {
	decoder: reed_solomon::ReedSolomonDecoder,
}

impl SubShardDecoder {
	pub fn new() -> Result<Self, Error> {
		Ok(Self {
			decoder: reed_solomon::ReedSolomonDecoder::new(
				N_SHARDS,
				N_REDUNDANCY * N_SHARDS,
				SHARD_MIN_SIZE * SUBSHARD_BATCH_MUL,
			)?,
		})
	}

	// u8 is the segment number.
	pub fn reconstruct<'a, I>(
		&mut self,
		subshards: &'a mut I,
	) -> Result<(Vec<(u8, Segment)>, usize), Error>
	where
		I: Iterator<Item = (u8, ChunkIndex, &'a SubShard)>,
	{
		let mut ori = vec![Vec::new(); TOTAL_SHARDS];
		let mut segments = BTreeSet::new();
		let mut nb_decode = 0;

		// TODO processed and run_segments could be skiped if we are sure to get
		// correct number of chunks all for the same given chunk ix and segments.
		let mut processed_segments = BTreeSet::new();
		for (segment, chunk_index, chunk) in subshards {
			ori[chunk_index.0 as usize].push((segment, chunk));
			segments.insert(segment);
		}

		let mut result = Vec::new();

		// Note that sometime byte could stay set to non zero value, but it does not matter.
		let mut shard = [0u8; SUBSHARD_BATCH_MUL * SHARD_MIN_SIZE];

		for segment in segments {
			if processed_segments.contains(&(segment as usize)) {
				continue;
			}
			let mut nb_chunk = 0;
			let mut ori_map: std::collections::BTreeMap<usize, Vec<u8>> = Default::default();
			// count all segments written, and stop at first segment having enough.
			let mut run_segments = BTreeMap::new();

			// Note this favor original chunks (firsts chunk_ix).
			for (chunk_ix, chunks) in ori.iter().enumerate() {
				if chunks.len() > 0 {
					let mut added = false;
					for (segment_i, chunk) in chunks {
						let segment_i = *segment_i as usize;
						if nb_chunk == 0 {
							if !processed_segments.contains(&segment_i) {
								run_segments.insert(segment_i, 1);
							} else {
								continue;
							}
						} else {
							if let Some(count) = run_segments.get_mut(&segment_i) {
								if *count == nb_chunk {
									*count += 1;
								} else {
									continue;
								}
							} else {
								continue;
							}
						}
						added = true;
						let shard_i_s = segment_i * SUBSHARD_SIZE / SHARD_MIN_SIZE;
						let shard_i_r = segment_i * SUBSHARD_SIZE % SHARD_MIN_SIZE;
						let mut shard_i = shard_i_s * SHARD_MIN_SIZE + shard_i_r / POINT_SIZE;
						for point_i in 0..SUBSHARD_POINTS {
							shard[shard_i] = chunk[point_i * POINT_SIZE];
							shard[shard_i + POINT_BYTE_SPACING] = chunk[(point_i * POINT_SIZE) + 1];
							shard_i += 1;
							if shard_i % POINT_BYTE_SPACING == 0 {
								shard_i += POINT_BYTE_SPACING;
							}
						}
					}
					if !added {
						continue;
					}
					if chunk_ix < N_SHARDS {
						self.decoder.add_original_shard(chunk_ix, &shard)?;
						ori_map.insert(chunk_ix, shard.to_vec());
					} else {
						self.decoder.add_recovery_shard(chunk_ix - N_SHARDS, &shard)?;
					}
					nb_chunk += 1;
					if nb_chunk == N_SHARDS {
						// we stop at first match, we cannot
						// attempt more as then we would not have matched completed
						// shards.
						break;
					}
				}
			}
			if nb_chunk != N_SHARDS {
				self.decoder.reset(
					N_SHARDS,
					N_REDUNDANCY * N_SHARDS,
					SHARD_MIN_SIZE * SUBSHARD_BATCH_MUL,
				)?;
				// none added: we did not have a single segment with enough shards.
				processed_segments.extend(run_segments.keys());
				continue;
			}
			let ori_ret = self.decoder.decode()?;
			nb_decode += 1;
			for (i, o) in ori_ret.restored_original_iter() {
				ori_map.insert(i, o.to_vec());
			}
			debug_assert_eq!(ori_map.len(), N_SHARDS);
			for segment in run_segments.iter().filter(|v| *v.1 == N_SHARDS).map(|v| *v.0) {
				let chunk_start = segment * SEGMENT_SIZE_ALIGNED;
				let original = ori_chunk_to_data(&ori_map, chunk_start, Some(SEGMENT_SIZE))
					.expect("number of segments checked");
				result.push((
					segment as u8,
					Segment { data: Box::new(original), index: segment as u32 },
				));
				processed_segments.insert(segment);
			}
		}
		Ok((result, nb_decode))
	}
}

fn ori_chunk_to_data(
	shards: &BTreeMap<usize, Vec<u8>>,
	start_data: usize,
	data_len: Option<usize>,
) -> Option<[u8; 4096]> {
	let mut data = [0u8; 4096];

	let mut i_data = 0;
	let (mut full_i, mut shard_i, mut shard_a) = data_index_to_chunk_index(start_data);
	let mut shard_i_offset = full_i * SHARD_MIN_SIZE;
	loop {
		let Some(s) = shards.get(&shard_a) else {
			return None;
		};
		let l = s[shard_i_offset + shard_i];
		data[i_data] = l;
		i_data += 1;
		let r = s[shard_i_offset + shard_i + POINT_BYTE_SPACING];
		data[i_data] = r;
		i_data += 1;
		if data_len.map(|m| i_data >= m).unwrap_or(false) {
			break;
		}
		shard_a += 1;
		if shard_a % N_SHARDS == 0 {
			shard_i += 1;
			if shard_i == POINT_BYTE_SPACING {
				shard_i = 0;
				full_i += 1;
				if full_i == SUBSHARD_BATCH_MUL {
					break;
				}

				shard_i_offset = full_i * SHARD_MIN_SIZE;
			}
			shard_a = 0;
		}
	}
	Some(data)
}

// return chunk index among N_SHARDS (group of n , ix in slice, ix in n)
fn data_index_to_chunk_index(index: usize) -> (usize, usize, usize) {
	let shard_batch_size = SHARD_MIN_SIZE * N_SHARDS;
	let a = index % shard_batch_size;
	let b = a % (N_SHARDS * POINT_SIZE);
	(index / shard_batch_size, a / (N_SHARDS * POINT_SIZE), b / POINT_SIZE)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_segments() {
		for count in [1, 2, 4, 5, 10, 16] {
			test_sc(count);
		}
	}

	fn test_sc(nb_seg: usize) {
		use rand::{rngs::SmallRng, Rng, SeedableRng};
		let mut rng = SmallRng::from_seed([0; 32]);
		let segments: Vec<_> = (0..nb_seg)
			.map(|s| {
				let mut se = [0u8; SEGMENT_SIZE];
				rng.fill::<_>(&mut se[..]);

				Segment { data: Box::new(se), index: s as u32 }
			})
			.collect();

		let mut encoder = SubShardEncoder::new().unwrap();
		let mut decoder = SubShardDecoder::new().unwrap();
		let chunks = encoder.construct_chunks(&segments).unwrap();

		for i_seg in 0..nb_seg {
			let mut it = (&chunks[i_seg][0..N_SHARDS / 3])
				.iter()
				.enumerate()
				.map(|(i, c)| (i_seg as u8, ChunkIndex(i as u16), c))
				.chain(
					(&chunks[i_seg][N_SHARDS..N_SHARDS + N_SHARDS / 3])
						.iter()
						.enumerate()
						.map(|(i, c)| (i_seg as u8, ChunkIndex(i as u16 + N_SHARDS as u16), c)),
				)
				.chain(
					(&chunks[i_seg][N_SHARDS * 2..N_SHARDS * 2 + N_SHARDS / 3])
						.iter()
						.enumerate()
						.map(|(i, c)| (i_seg as u8, ChunkIndex(i as u16 + N_SHARDS as u16 * 2), c)),
				);
			let (s, i) = decoder.reconstruct(&mut it).unwrap();
			assert_eq!(i, 1);
			assert_eq!((i_seg as u8, segments[i_seg].clone()), s[0]);
		}
		// try batching 2 subchunk
		if nb_seg < 2 {
			return;
		}
		let i_seg1 = 0;
		let i_seg2 = nb_seg / 2;

		let it1 = (&chunks[i_seg1][0..N_SHARDS / 3])
			.iter()
			.enumerate()
			.map(|(i, c)| (i_seg1 as u8, ChunkIndex(i as u16), c))
			.chain(
				(&chunks[i_seg1][N_SHARDS..N_SHARDS + N_SHARDS / 3])
					.iter()
					.enumerate()
					.map(|(i, c)| (i_seg1 as u8, ChunkIndex(i as u16 + N_SHARDS as u16), c)),
			)
			.chain(
				(&chunks[i_seg1][N_SHARDS * 2..N_SHARDS * 2 + N_SHARDS / 3])
					.iter()
					.enumerate()
					.map(|(i, c)| (i_seg1 as u8, ChunkIndex(i as u16 + N_SHARDS as u16 * 2), c)),
			);
		let it2 = (&chunks[i_seg2][0..N_SHARDS / 3])
			.iter()
			.enumerate()
			.map(|(i, c)| (i_seg2 as u8, ChunkIndex(i as u16), c))
			.chain(
				(&chunks[i_seg2][N_SHARDS..N_SHARDS + N_SHARDS / 3])
					.iter()
					.enumerate()
					.map(|(i, c)| (i_seg2 as u8, ChunkIndex(i as u16 + N_SHARDS as u16), c)),
			)
			.chain(
				(&chunks[i_seg2][N_SHARDS * 2..N_SHARDS * 2 + N_SHARDS / 3])
					.iter()
					.enumerate()
					.map(|(i, c)| (i_seg2 as u8, ChunkIndex(i as u16 + N_SHARDS as u16 * 2), c)),
			);

		let (s, i) = decoder.reconstruct(&mut it1.chain(it2)).unwrap();
		assert_eq!(i, 1); // all chunk ix are aligned so can be processed at once.
		assert_eq!((i_seg1 as u8, segments[i_seg1].clone()), s[0]);
		assert_eq!((i_seg2 as u8, segments[i_seg2].clone()), s[1]);

		let it1 = (&chunks[i_seg1][0..N_SHARDS / 3])
			.iter()
			.enumerate()
			.map(|(i, c)| (i_seg1 as u8, ChunkIndex(i as u16), c))
			.chain(
				(&chunks[i_seg1][N_SHARDS..N_SHARDS + N_SHARDS / 3])
					.iter()
					.enumerate()
					.map(|(i, c)| (i_seg1 as u8, ChunkIndex(i as u16 + N_SHARDS as u16), c)),
			)
			.chain(
				(&chunks[i_seg1][N_SHARDS * 2..N_SHARDS * 2 + N_SHARDS / 3])
					.iter()
					.enumerate()
					.map(|(i, c)| (i_seg1 as u8, ChunkIndex(i as u16 + N_SHARDS as u16 * 2), c)),
			);

		// unaligned a batch of chunks
		let it3 = (&chunks[i_seg2][0 + 1..N_SHARDS / 3 + 1])
			.iter()
			.enumerate()
			.map(|(i, c)| (i_seg2 as u8, ChunkIndex(i as u16 + 1), c))
			.chain(
				(&chunks[i_seg2][N_SHARDS..N_SHARDS + N_SHARDS / 3])
					.iter()
					.enumerate()
					.map(|(i, c)| (i_seg2 as u8, ChunkIndex(i as u16 + N_SHARDS as u16), c)),
			)
			.chain(
				(&chunks[i_seg2][N_SHARDS * 2..N_SHARDS * 2 + N_SHARDS / 3])
					.iter()
					.enumerate()
					.map(|(i, c)| (i_seg2 as u8, ChunkIndex(i as u16 + N_SHARDS as u16 * 2), c)),
			);
		let (s, i) = decoder.reconstruct(&mut it1.chain(it3)).unwrap();
		assert_eq!(i, 2); // not all chunk ix are aligned
		assert_eq!((i_seg1 as u8, segments[i_seg1].clone()), s[0]);
		assert_eq!((i_seg2 as u8, segments[i_seg2].clone()), s[1]);
	}
}
