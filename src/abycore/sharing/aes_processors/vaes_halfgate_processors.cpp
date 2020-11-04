#include "vaes_halfgate_processors.h"
#include "../yaosharing.h"

#include <wmmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <iostream>
#include <iomanip>
#include <algorithm>

// in number of tables
constexpr size_t mainGarblingWidthVaes = 4;
constexpr size_t mainEvaluatingWidthVaes = 8;

void FixedKeyLTEvaluatingVaesProcessor::computeAESPreOutKeys(uint32_t tableCounter, size_t numTablesInBatch)
{
	if (m_gateQueue.size() == 0)
		return;

	const size_t leftovers = numTablesInBatch % mainEvaluatingWidthVaes;
	const size_t mainBulkSize = numTablesInBatch - leftovers;

	computeAESPreOutKeys<mainEvaluatingWidthVaes>(tableCounter, 0, 0, mainBulkSize);

	size_t numTablesLeft = 0;
	size_t ridx;

	for (ridx = m_gateQueue.size() - 1; ridx >= 0; --ridx)
	{
		numTablesLeft += m_gateQueue[ridx]->nvals;
		if (numTablesLeft >= leftovers)
			break;
	}

	if (leftovers > 0)
	{
		computeAESPreOutKeys<1>(tableCounter + mainBulkSize, ridx, numTablesLeft - leftovers, leftovers);
	}
}

void FixedKeyLTGarblingVaesProcessor::fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch)
{
	const size_t leftovers = numTablesInBatch % mainGarblingWidthVaes;
	const size_t mainBulkSize = numTablesInBatch - leftovers;

	fillAESBufferAND<mainGarblingWidthVaes>(baseOffset, tableCounter, mainBulkSize, 0);

	if (leftovers > 0)
	{
		fillAESBufferAND<1>(baseOffset + mainBulkSize, tableCounter + mainBulkSize, leftovers, mainBulkSize * 16 * 4); // 16 bytes per ciphertext, 4 per table
	}
}

template<size_t width>
inline void FixedKeyLTEvaluatingVaesProcessor::computeAESPreOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch)
{
	constexpr size_t div_width = (width + 3) / 4; // ceiling division

	static_assert((width < 4) || (width % 4 == 0));

	const __m512i ONE = _mm512_set_epi32(
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 0, 1);

	// the offset calculation WILL break down if the above static assert is violated
	// Example: width==7
	// then the first 4 would need an offset of 8
	// and then the second 3 would need an offset of 6 so the next round has the right offset
	// we *could* fix this with a div_width array of offsets where the first div_width-1 elements have the 8
	// and the remainder has (width%4)*2 however we don't need such weird instances
	constexpr size_t offset = std::min(size_t(4) * KEYS_PER_GATE_IN_TABLE, width * KEYS_PER_GATE_IN_TABLE);

	const __m512i FULL_OFFSET = _mm512_set_epi32(
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset);



	__m512i counter = _mm512_set_epi32(
		0, 0, 0, (tableCounter + 3) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 2) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 1) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 0) * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i leftData[div_width];
	__m512i rightData[div_width];
	__m512i leftKeys[div_width];
	__m512i rightKeys[div_width];
	uint8_t* targetGateKey[width];
	__m512i aes_keys[11];

	for (size_t i = 0; i < 11; ++i)
	{
		__m128i temp_key = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
		aes_keys[i] = _mm512_broadcast_i32x4(temp_key);
	}

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		__m128i leftTemp[width];
		__m128i rightTemp[width];
		// TODO: optimize using bigger vector loads potentially?
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];

			leftTemp[w] = _mm_loadu_si128((__m128i*)(leftParent->gs.yval + 16 * currentOffset));
			rightTemp[w] = _mm_loadu_si128((__m128i*)(rightParent->gs.yval + 16 * currentOffset));

			targetGateKey[w] = currentGate->gs.yval + 16 * currentOffset;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

#define INSERT(target,source,offset)\
		if(w*4+offset>=width) {break;}\
		target[w] = _mm512_inserti32x4(target[w],source[4*w+offset],offset)

		// need separate loops here or else the break of the left insertions would skip the right ones
		for (size_t w = 0; w < div_width; ++w)
		{
			//INSERT(leftKeys, leftTemp, 0);
			leftKeys[w] = _mm512_castsi128_si512(leftTemp[4 * w]);
			INSERT(leftKeys, leftTemp, 1);
			INSERT(leftKeys, leftTemp, 2);
			INSERT(leftKeys, leftTemp, 3);
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			//INSERT(rightKeys, rightTemp, 0);
			rightKeys[w] = _mm512_castsi128_si512(rightTemp[4 * w]);
			INSERT(rightKeys, rightTemp, 1);
			INSERT(rightKeys, rightTemp, 2);
			INSERT(rightKeys, rightTemp, 3);
		}
#undef INSERT

		for (size_t w = 0; w < div_width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			leftData[w] = counter;
			rightData[w] = _mm512_add_epi32(counter, ONE);
			counter = _mm512_add_epi32(counter, FULL_OFFSET);

		}

		for (size_t w = 0; w < div_width; ++w)
		{
			// this assumes that we actually use the correct constant that sets the top bit
			// this is the left shift by 1 bit
			__m512i tempL = _mm512_slli_epi64(leftKeys[w], 1);
			__m512i tempR = _mm512_srli_epi64(leftKeys[w], 63);
			tempR = _mm512_shuffle_epi32(tempR, 0x4E); // 0x4E is 01 00 11 10 in binary which is exactly a 64-bit word lane swap
			const __m512i topExtractor = _mm512_set_epi64(
				~0, 0,
				~0, 0,
				~0, 0,
				~0, 0);
			__m512i topBit = _mm512_and_si512(tempR, topExtractor);
			leftKeys[w] = _mm512_xor_si512(tempL, topBit);

			tempL = _mm512_slli_epi64(rightKeys[w], 1);
			tempR = _mm512_srli_epi64(rightKeys[w], 63);
			tempR = _mm512_shuffle_epi32(tempR, 0x4E); // 0x4E is 01 00 11 10 in binary which is exactly a 64-bit word lane swap
			topBit = _mm512_and_si512(tempR, topExtractor);
			rightKeys[w] = _mm512_xor_si512(tempL, topBit);

			//parentKeys[w] = _mm_slli_si128(parentKeys[w], 1); // this does BYTE shift not BIT shifts!1!

			// this is the actual AES input
			leftData[w] = _mm512_xor_si512(leftData[w], leftKeys[w]);
			rightData[w] = _mm512_xor_si512(rightData[w], rightKeys[w]);

			leftKeys[w] = leftData[w]; // keep as a backup for post-whitening
			rightKeys[w] = rightData[w]; // keep as a backup for post-whitening
			leftData[w] = _mm512_xor_si512(leftData[w], aes_keys[0]);
			rightData[w] = _mm512_xor_si512(rightData[w], aes_keys[0]);
		}

		for (size_t r = 1; r < 10; ++r)
		{
			for (size_t w = 0; w < div_width; ++w)
			{
				leftData[w] = _mm512_aesenc_epi128(leftData[w], aes_keys[r]);
				rightData[w] = _mm512_aesenc_epi128(rightData[w], aes_keys[r]);
			}
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			leftData[w] = _mm512_aesenclast_epi128(leftData[w], aes_keys[10]);
			rightData[w] = _mm512_aesenclast_epi128(rightData[w], aes_keys[10]);
			leftData[w] = _mm512_xor_si512(leftData[w], leftKeys[w]);
			rightData[w] = _mm512_xor_si512(rightData[w], rightKeys[w]);
			leftData[w] = _mm512_xor_si512(leftData[w], rightData[w]);
		}


#define EXTRACT_AND_STORE(l) \
		if(w*4+l>=width) {break;}\
		extracted = _mm512_extracti32x4_epi32(leftData[w], l);\
		_mm_storeu_si128((__m128i*)(targetGateKey[w*4+l]), extracted)
		for (size_t w = 0; w < div_width; ++w)
		{
			__m128i extracted;
			extracted = _mm512_extracti32x4_epi32(leftData[w], 0);
			_mm_storeu_si128((__m128i*)(targetGateKey[w * 4]), extracted);
			//EXTRACT_AND_STORE(0);
			EXTRACT_AND_STORE(1);
			EXTRACT_AND_STORE(2);
			EXTRACT_AND_STORE(3);
		}
#undef EXTRACT_AND_STORE
	}
}

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void FixedKeyLTGarblingVaesProcessor::fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch, size_t bufferOffset)
{
	assert(bufferOffset + numTablesInBatch * 4 * 16 <= m_bufferSize);

	std::cout << "garbling VAES" << std::endl;

	const __m512i COUNTER_DIFF = _mm512_set_epi32(
		0, 0, 0, 2,
		0, 0, 0, 2,
		0, 0, 0, 2,
		0, 0, 0, 2
	);

	__m512i counter = _mm512_set_epi32(
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 1,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 1,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i data[width];
	__m512i keys[width];
	__m512i aes_keys[11];

	for (size_t i = 0; i < 11; ++i)
	{
		__m128i temp_key = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
		aes_keys[i] = _mm512_broadcast_i32x4(temp_key);
	}

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize this to detect and exploit simd gates
		// saving vGate, parent base pointer and owning gate lookups
		for (size_t w = 0; w < width; ++w)
		{
			const size_t index = w + i + baseOffset;
			const auto currentGate = m_tableGateQueue[index];
			const uint32_t leftParentId = currentGate.owningGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate.owningGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint32_t simdOffset = currentGate.simdPosition;

			__m128i lowerLow = _mm_loadu_si128((__m128i*)(leftParent->gs.yinput.outKey + 16 * simdOffset));
			__m128i upperLow = _mm_xor_si128(lowerLow, R);
			__m128i lowerUpper = _mm_loadu_si128((__m128i*)(rightParent->gs.yinput.outKey + 16 * simdOffset));
			__m128i upperUpper = _mm_xor_si128(lowerUpper, R);

			// this is a "tree" construction due to the high latency of the inserti instructions
			keys[w] = _mm512_castsi128_si512(lowerLow);
			__m256i upper = _mm256_castsi128_si256(lowerUpper);
			upper = _mm256_inserti128_si256(upper, upperUpper, 1);
			keys[w] = _mm512_inserti32x4(keys[w], upperLow, 1);
			keys[w] = _mm512_inserti64x4(keys[w], upper, 1);
		}

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = counter;
			counter = _mm512_add_epi32(counter, COUNTER_DIFF);
		}

		for (size_t w = 0; w < width; ++w)
		{
			// this assumes that we actually use the correct constant that sets the top bit
			// this is the left shift by 1 bit
			__m512i tempL = _mm512_slli_epi64(keys[w], 1);
			__m512i tempR = _mm512_srli_epi64(keys[w], 63);
			tempR = _mm512_shuffle_epi32(tempR, 0x4E); // 0x4E is 01 00 11 10 in binary which is exactly a 64-bit word lane swap, note that this instruction is within 128-bit lanes
			__m512i topExtractor = _mm512_set_epi64(
				~0, 0,
				~0, 0,
				~0, 0,
				~0, 0);
			__m512i topBit = _mm512_and_si512(tempR, topExtractor);
			keys[w] = _mm512_xor_si512(tempL, topBit);

			//parentKeys[w] = _mm_slli_si128(parentKeys[w], 1); // this does BYTE shift not BIT shifts!1!

			// this is the actual AES input
			data[w] = _mm512_xor_si512(data[w], keys[w]);

			keys[w] = data[w]; // keep as a backup for post-whitening
			data[w] = _mm512_xor_si512(data[w], aes_keys[0]);
		}

		for (size_t r = 1; r < 10; ++r)
		{
			for (size_t w = 0; w < width; ++w)
			{
				data[w] = _mm512_aesenc_epi128(data[w], aes_keys[r]);
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = _mm512_aesenclast_epi128(data[w], aes_keys[10]);
			data[w] = _mm512_xor_si512(data[w], keys[w]);
		}


		for (size_t w = 0; w < width; ++w)
			_mm512_store_si512((__m512i*)(m_aesBuffer + bufferOffset + (i + w) * 64), data[w]); // 64 = 4 (#PRF calls) * 16 (PRF result size)
	}
}