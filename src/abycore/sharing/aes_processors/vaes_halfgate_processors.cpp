#include "vaes_halfgate_processors.h"
#include "../yaosharing.h"

#include <wmmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <iostream>
#include <iomanip>
#include <algorithm>

// this is a trick from Rust:
// assumes that this function will be inlined *and*
// that the loop variable which is fed into location will be unrolled
static inline __m512i mm512_insert_128(__m512i baseline, __m128i word, size_t location) {
	switch (location & 0x03)
	{
	case 0:
		return _mm512_inserti32x4(baseline, word, 0);
	case 1:
		return _mm512_inserti32x4(baseline, word, 1);
	case 2:
		return _mm512_inserti32x4(baseline, word, 2);
	case 3:
		return _mm512_inserti32x4(baseline, word, 3);
	}
}

// this is a trick from Rust:
// assumes that this function will be inlined *and*
// that the loop variable which is fed into location will be unrolled
static inline __m128i mm512_extract_128(__m512i baseline, size_t location) {
	switch (location & 0x03)
	{
	case 0:
		return _mm512_extracti32x4_epi32(baseline, 0);
	case 1:
		return _mm512_extracti32x4_epi32(baseline, 1);
	case 2:
		return _mm512_extracti32x4_epi32(baseline, 2);
	case 3:
		return _mm512_extracti32x4_epi32(baseline, 3);
	}
}

// in number of tables
constexpr size_t mainGarblingWidthPRPVaes = 8;
constexpr size_t mainEvaluatingWidthPRPVaes = 16;
constexpr size_t mainGarblingWidthCircVaes = 6;
constexpr size_t mainEvaluatingWidthCircVaes = 16;

static void PrintKey(__m512i data) {
	uint8_t key[64];
	_mm512_storeu_si512((__m512i*)key, data);

	for (int j = 0; j < 64; j += 16)
	{
		for (uint32_t i = 0; i < 16; i++) {
			std::cout << std::setw(2) << std::setfill('0') << (std::hex) << (uint32_t)key[i + j];
		}
		std::cout << std::endl;
	}

	std::cout << (std::dec);
}

static void PrintKey(__m128i data) {
	uint8_t key[16];
	_mm_storeu_si128((__m128i*)key, data);

	for (uint32_t i = 0; i < 16; i++) {
		std::cout << std::setw(2) << std::setfill('0') << (std::hex) << (uint32_t)key[i];
	}
	std::cout << (std::dec);
}

void FixedKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t FixedKeyLTEvaluatingVaesProcessor::vectorWidth() const
{
	return mainEvaluatingWidthPRPVaes;
}

void FixedKeyLTEvaluatingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthPRPVaes>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyLTEvaluatingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyLTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t FixedKeyLTGarblingVaesProcessor::vectorWidth() const
{
	return mainGarblingWidthPRPVaes;
}

void FixedKeyLTGarblingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthPRPVaes>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void FixedKeyLTGarblingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void InputKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t InputKeyLTEvaluatingVaesProcessor::vectorWidth() const
{
	return mainEvaluatingWidthCircVaes;
}

void InputKeyLTEvaluatingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthCircVaes>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void InputKeyLTEvaluatingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void InputKeyLTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t InputKeyLTGarblingVaesProcessor::vectorWidth() const
{
	return mainGarblingWidthCircVaes;
}

void InputKeyLTGarblingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthCircVaes>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void InputKeyLTGarblingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}


template<size_t width>
inline void FixedKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t div_width = (width + 3) / 4; // ceiling division
	constexpr size_t num_buffer_words = std::min(width, size_t(4));

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
	__m512i finalMask[div_width];
	uint8_t* targetGateKey[width];
	__m512i aes_keys[11];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;

	for (size_t i = 0; i < 11; ++i)
	{
		__m128i temp_key = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
		aes_keys[i] = _mm512_broadcast_i32x4(temp_key);
	}

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize using bigger vector loads potentially?
		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const GATE* currentGate = m_gateQueue[currentGateIdx];
				const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
				const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
				const GATE* leftParent = &m_vGates[leftParentId];
				const GATE* rightParent = &m_vGates[rightParentId];
				const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
				const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

				const __m128i leftParentKeyLocal = _mm_loadu_si128((__m128i*)leftParentKey);
				leftKeys[w] = mm512_insert_128(leftKeys[w], leftParentKeyLocal, k);
				const __m128i rightParentKeyLocal = _mm_loadu_si128((__m128i*)rightParentKey);
				rightKeys[w] = mm512_insert_128(rightKeys[w], rightParentKeyLocal, k);

				targetGateKey[4*w+k] = currentGate->gs.yval + 16 * currentOffset;

				const uint8_t lpbit = leftParentKey[15] & 0x01;
				const uint8_t lpbit11 = (lpbit << 1) | lpbit;
				const uint8_t rpbit = rightParentKey[15] & 0x01;
				const uint8_t rpbit11 = (rpbit << 1) | rpbit;

				__m128i finalMaskLocal = _mm_maskz_loadu_epi64(lpbit11, (__m128i*)gtptr);
				gtptr += 16;
				const __m128i rightTable = _mm_loadu_si128((__m128i*)gtptr);
				const __m128i rightMaskUpdate = _mm_xor_si128(rightTable, leftParentKeyLocal);
				finalMaskLocal = _mm_mask_xor_epi64(finalMaskLocal, rpbit11, finalMaskLocal, rightMaskUpdate);
				gtptr += 16;

				finalMask[w] = mm512_insert_128(finalMask[w], finalMaskLocal, k);

				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			leftData[w] = counter;
			rightData[w] = _mm512_add_epi32(counter, ONE);
			counter = _mm512_add_epi32(counter, FULL_OFFSET);

		}

		for (size_t w = 0; w < div_width; ++w)
		{
			// this is the 128-bit leftshift code from https://stackoverflow.com/a/34482688/4733946
			// as requested by User0 https://stackoverflow.com/users/5720018/user0
			// and given by Peter Cordes https://stackoverflow.com/users/224132/peter-cordes

			__m512i leftCarry = _mm512_bslli_epi128(leftKeys[w], 8);
			leftCarry = _mm512_srli_epi64(leftCarry, 63);
			leftKeys[w] = _mm512_slli_epi64(leftKeys[w], 1);
			leftKeys[w] = _mm512_or_si512(leftKeys[w], leftCarry);

			__m512i rightCarry = _mm512_bslli_epi128(rightKeys[w], 8);
			rightCarry = _mm512_srli_epi64(rightCarry, 63);
			rightKeys[w] = _mm512_slli_epi64(rightKeys[w], 1);
			rightKeys[w] = _mm512_or_si512(rightKeys[w], rightCarry);

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
			leftData[w] = _mm512_xor_si512(leftData[w], finalMask[w]);
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const __m128i extracted = mm512_extract_128(leftData[w],k);
				_mm_storeu_si128((__m128i*)(targetGateKey[4 * w + k]), extracted);
			}
		}
	}
}

template<size_t width>
inline void InputKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t div_width = (width + 3) / 4; // ceiling division
	constexpr size_t num_buffer_words = std::min(width, size_t(4));

	static_assert((width < 4) || (width % 4 == 0), "This implementation only supports multiplies of 4 or values smaller than 4.");

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
	__m512i finalMask[div_width];
	uint8_t* targetGateKey[width];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize using bigger vector loads potentially?
		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const GATE* currentGate = m_gateQueue[currentGateIdx];
				const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
				const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
				const GATE* leftParent = &m_vGates[leftParentId];
				const GATE* rightParent = &m_vGates[rightParentId];
				const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
				const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

				const __m128i leftParentKeyLocal = _mm_loadu_si128((__m128i*)leftParentKey);
				leftKeys[w] = mm512_insert_128(leftKeys[w], leftParentKeyLocal, k);
				const __m128i rightParentKeyLocal = _mm_loadu_si128((__m128i*)rightParentKey);
				rightKeys[w] = mm512_insert_128(rightKeys[w], rightParentKeyLocal, k);

				targetGateKey[4 * w + k] = currentGate->gs.yval + 16 * currentOffset;

				const uint8_t lpbit = leftParentKey[15] & 0x01;
				const uint8_t lpbit11 = (lpbit << 1) | lpbit;
				const uint8_t rpbit = rightParentKey[15] & 0x01;
				const uint8_t rpbit11 = (rpbit << 1) | rpbit;

				__m128i finalMaskLocal = _mm_maskz_loadu_epi64(lpbit11, (__m128i*)gtptr);
				gtptr += 16;
				const __m128i rightTable = _mm_loadu_si128((__m128i*)gtptr);
				const __m128i rightMaskUpdate = _mm_xor_si128(rightTable, leftParentKeyLocal);
				finalMaskLocal = _mm_mask_xor_epi64(finalMaskLocal, rpbit11, finalMaskLocal, rightMaskUpdate);
				gtptr += 16;

				finalMask[w] = mm512_insert_128(finalMask[w], finalMaskLocal, k);

				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			leftData[w] = counter;
			rightData[w] = _mm512_add_epi32(counter, ONE);
			counter = _mm512_add_epi32(counter, FULL_OFFSET);

		}

		for (size_t w = 0; w < div_width; ++w)
		{
			leftData[w] = _mm512_xor_si512(leftData[w], leftKeys[w]);
			rightData[w] = _mm512_xor_si512(rightData[w], rightKeys[w]);
		}

		__m512i rcon = _mm512_set1_epi32(1);
		const __m512i shuffle_mask = _mm512_set1_epi32(0x0c0f0e0d);
		const __m512i rcon_multiplier = _mm512_set1_epi8(2);
		__m512i temp2[2*div_width], temp3[2*div_width];

		for (size_t r = 1; r < 10; ++r)
		{
			for (size_t w = 0; w < div_width; ++w)
			{
				temp2[2*w+0] = _mm512_shuffle_epi8(leftKeys[w], shuffle_mask);
				temp2[2*w+0] = _mm512_aesenclast_epi128(temp2[2*w+0], rcon);
				// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
				temp3[2*w+0] = _mm512_bslli_epi128(leftKeys[w], 0x4);
				leftKeys[w] = _mm512_xor_si512(leftKeys[w], temp3[2*w+0]);
				temp3[2*w+0] = _mm512_bslli_epi128(temp3[2*w+0], 0x4);
				leftKeys[w] = _mm512_xor_si512(leftKeys[w], temp3[2*w+0]);
				temp3[2*w+0] = _mm512_bslli_epi128(temp3[2*w+0], 0x4);
				leftKeys[w] = _mm512_xor_si512(leftKeys[w], temp3[2*w+0]);
				leftKeys[w] = _mm512_xor_si512(leftKeys[w], temp2[2*w+0]);

				temp2[2 * w + 1] = _mm512_shuffle_epi8(rightKeys[w], shuffle_mask);
				temp2[2 * w + 1] = _mm512_aesenclast_epi128(temp2[2 * w + 1], rcon);
				// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
				temp3[2 * w + 1] = _mm512_bslli_epi128(rightKeys[w], 0x4);
				rightKeys[w] = _mm512_xor_si512(rightKeys[w], temp3[2 * w + 1]);
				temp3[2 * w + 1] = _mm512_bslli_epi128(temp3[2 * w + 1], 0x4);
				rightKeys[w] = _mm512_xor_si512(rightKeys[w], temp3[2 * w + 1]);
				temp3[2 * w + 1] = _mm512_bslli_epi128(temp3[2 * w + 1], 0x4);
				rightKeys[w] = _mm512_xor_si512(rightKeys[w], temp3[2 * w + 1]);
				rightKeys[w] = _mm512_xor_si512(rightKeys[w], temp2[2 * w + 1]);


				leftData[w] = _mm512_aesenc_epi128(leftData[w], leftKeys[w]);
				rightData[w] = _mm512_aesenc_epi128(rightData[w], rightKeys[w]);
			}

			rcon = _mm512_gf2p8mul_epi8(rcon, rcon_multiplier);
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			temp2[2 * w + 0] = _mm512_shuffle_epi8(leftKeys[w], shuffle_mask);
			temp2[2 * w + 0] = _mm512_aesenclast_epi128(temp2[2 * w + 0], rcon);
			// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
			temp3[2 * w + 0] = _mm512_bslli_epi128(leftKeys[w], 0x4);
			leftKeys[w] = _mm512_xor_si512(leftKeys[w], temp3[2 * w + 0]);
			temp3[2 * w + 0] = _mm512_bslli_epi128(temp3[2 * w + 0], 0x4);
			leftKeys[w] = _mm512_xor_si512(leftKeys[w], temp3[2 * w + 0]);
			temp3[2 * w + 0] = _mm512_bslli_epi128(temp3[2 * w + 0], 0x4);
			leftKeys[w] = _mm512_xor_si512(leftKeys[w], temp3[2 * w + 0]);
			leftKeys[w] = _mm512_xor_si512(leftKeys[w], temp2[2 * w + 0]);

			temp2[2 * w + 1] = _mm512_shuffle_epi8(rightKeys[w], shuffle_mask);
			temp2[2 * w + 1] = _mm512_aesenclast_epi128(temp2[2 * w + 1], rcon);
			// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
			temp3[2 * w + 1] = _mm512_bslli_epi128(rightKeys[w], 0x4);
			rightKeys[w] = _mm512_xor_si512(rightKeys[w], temp3[2 * w + 1]);
			temp3[2 * w + 1] = _mm512_bslli_epi128(temp3[2 * w + 1], 0x4);
			rightKeys[w] = _mm512_xor_si512(rightKeys[w], temp3[2 * w + 1]);
			temp3[2 * w + 1] = _mm512_bslli_epi128(temp3[2 * w + 1], 0x4);
			rightKeys[w] = _mm512_xor_si512(rightKeys[w], temp3[2 * w + 1]);
			rightKeys[w] = _mm512_xor_si512(rightKeys[w], temp2[2 * w + 1]);

			leftData[w] = _mm512_aesenclast_epi128(leftData[w], leftKeys[w]);
			rightData[w] = _mm512_aesenclast_epi128(rightData[w], rightKeys[w]);

			leftData[w] = _mm512_xor_si512(leftData[w], rightData[w]);
			leftData[w] = _mm512_xor_si512(leftData[w], finalMask[w]);
		}


		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const __m128i extracted = mm512_extract_128(leftData[w], k);
				_mm_storeu_si128((__m128i*)(targetGateKey[4 * w + k]), extracted);
			}
		}
	}
}

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void FixedKeyLTGarblingVaesProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
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
	__m512i postMask[width];
	uint8_t* targetGateKey[width];
	uint8_t* targetGateKeyR[width];
	uint8_t rpbit[width];
	uint8_t finalMask[width];
	uint8_t* targetPiBit[width];

	for (size_t i = 0; i < 11; ++i)
	{
		__m128i temp_key = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
		aes_keys[i] = _mm512_broadcast_i32x4(temp_key);
	}

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);
	const __m512i wideR = _mm512_broadcast_i32x4(R);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize this to detect and exploit simd gates
		// saving vGate, parent base pointer and owning gate lookups
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_tableGateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t lpbit = leftParent->gs.yinput.pi[currentOffset];
			const uint8_t lpbit11 = (lpbit << 1) | lpbit;
			const uint8_t rpbitLocal = rightParent->gs.yinput.pi[currentOffset];
			const uint8_t rpbit11 = (rpbitLocal << 1) | rpbitLocal;
			rpbit[w] = rpbit11;

			currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbitLocal;

			const __m128i lowerLow = _mm_loadu_si128((__m128i*)leftParentKey);
			const __m128i upperLow = _mm_xor_si128(lowerLow, R);
			const __m128i lowerUpper = _mm_loadu_si128((__m128i*)rightParentKey);
			const __m128i upperUpper = _mm_xor_si128(lowerUpper, R);

			const __m128i toBeInserted = _mm_mask_blend_epi64(lpbit11, lowerLow, upperLow);
			postMask[w] = _mm512_inserti32x4(wideR, toBeInserted, 2);

			/*if (lpbit)
				postMask[w] = _mm512_inserti32x4(wideR, upperLow, 2);
			else
				postMask[w] = _mm512_inserti32x4(wideR, lowerLow, 2);*/

			targetGateKey[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKeyR[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;
			targetPiBit[w] = currentGate->gs.yinput.pi + currentOffset;

			const uint8_t lsbitANDrsbit = (leftParentKey[15] & 0x01) & (rightParentKey[15] & 0x01);
			const uint8_t lsbitANDrsbit11 = (lsbitANDrsbit << 1) | lsbitANDrsbit;

			finalMask[w] = lsbitANDrsbit11 | (rpbit11 << 2) | (0x03 << 4) | (lsbitANDrsbit11 << 6);

			// this is a "tree" construction due to the high latency of the inserti instructions
			keys[w] = _mm512_castsi128_si512(lowerLow);
			__m256i upper = _mm256_castsi128_si256(lowerUpper);
			upper = _mm256_inserti128_si256(upper, upperUpper, 1);
			keys[w] = _mm512_inserti32x4(keys[w], upperLow, 1);
			keys[w] = _mm512_inserti64x4(keys[w], upper, 1);

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = counter;
			counter = _mm512_add_epi32(counter, COUNTER_DIFF);
		}

		for (size_t w = 0; w < width; ++w)
		{
			// this is the 128-bit leftshift code from https://stackoverflow.com/a/34482688/4733946
			// as requested by User0 https://stackoverflow.com/users/5720018/user0
			// and given by Peter Cordes https://stackoverflow.com/users/224132/peter-cordes

			__m512i carry = _mm512_bslli_epi128(keys[w], 8);
			carry = _mm512_srli_epi64(carry, 63);
			keys[w] = _mm512_slli_epi64(keys[w], 1);
			keys[w] = _mm512_or_si512(keys[w], carry);

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
		{
			// intent (shuffling into, in terms of 128-bit lanes):
			// 2
			// 0
			// 3
			// 0
			const __m512i shuffleKey = _mm512_set_epi64(
				1, 0,
				7, 6,
				1, 0,
				5, 4
			);
			__m512i shuffledCopy = _mm512_permutexvar_epi64(shuffleKey, data[w]);
			__m512i firstXor = _mm512_xor_si512(data[w], shuffledCopy);

			__m512i secondXor = _mm512_mask_xor_epi64(firstXor, finalMask[w], firstXor, postMask[w]);
			_mm_storeu_si128((__m128i*)gtptr, _mm512_extracti32x4_epi32(secondXor, 1));
			gtptr += 16;
			_mm_storeu_si128((__m128i*)gtptr, _mm512_extracti32x4_epi32(secondXor, 2));
			gtptr += 16;

			const __m128i nonRpbitOutKey = _mm512_extracti32x4_epi32(secondXor, 0);
			const __m128i rpbitRXor = _mm512_extracti32x4_epi32(firstXor, 2);
			const __m128i rpbitLXor = _mm512_extracti32x4_epi32(secondXor, 3);
			__m128i outKey = _mm_mask_xor_epi64(nonRpbitOutKey, rpbit[w], rpbitLXor, rpbitRXor);

			/*
			if (rpbit[w]) {
				__m128i rXor = _mm512_extracti32x4_epi32(firstXor, 2);
				outKey = _mm512_extracti32x4_epi32(secondXor, 3);
				outKey = _mm_xor_si128(outKey, rXor);
			}
			else {
				outKey = _mm512_extracti32x4_epi32(secondXor, 0);
			}
			*/
			//uint8_t rBit = _mm_extract_epi8(R, 15) & 0x01;
			//assert(rBit == 1);
			uint8_t outWireBit = _mm_extract_epi8(outKey, 15) & 0x01;
			*targetPiBit[w] ^= outWireBit;
			const uint8_t amplifiedOutWireBit = (outWireBit << 1) | outWireBit;
			outKey = _mm_mask_xor_epi64(outKey, amplifiedOutWireBit, outKey, R);
			/*if (outWireBit) {
				outKey = _mm_xor_si128(outKey, R);
				*targetPiBit[w] ^= rBit;
			}*/
				
			_mm_storeu_si128((__m128i*)targetGateKey[w], outKey);
			_mm_storeu_si128((__m128i*)targetGateKeyR[w], _mm_xor_si128(outKey,R));
		}
	}
}

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void InputKeyLTGarblingVaesProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
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
	__m512i postMask[width];
	uint8_t* targetGateKey[width];
	uint8_t* targetGateKeyR[width];
	uint8_t rpbit[width];
	uint8_t finalMask[width];
	uint8_t* targetPiBit[width];

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);
	const __m512i wideR = _mm512_broadcast_i32x4(R);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize this to detect and exploit simd gates
		// saving vGate, parent base pointer and owning gate lookups
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_tableGateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t lpbit = leftParent->gs.yinput.pi[currentOffset];
			const uint8_t lpbit11 = (lpbit << 1) | lpbit;
			const uint8_t rpbitLocal = rightParent->gs.yinput.pi[currentOffset];
			const uint8_t rpbit11 = (rpbitLocal << 1) | rpbitLocal;
			rpbit[w] = rpbit11;

			currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbit[w];

			const __m128i lowerLow = _mm_loadu_si128((__m128i*)leftParentKey);
			const __m128i upperLow = _mm_xor_si128(lowerLow, R);
			const __m128i lowerUpper = _mm_loadu_si128((__m128i*)rightParentKey);
			const __m128i upperUpper = _mm_xor_si128(lowerUpper, R);

			const __m128i toBeInserted = _mm_mask_blend_epi64(lpbit11, lowerLow, upperLow);
			postMask[w] = _mm512_inserti32x4(wideR, toBeInserted, 2);

			/*if (lpbit)
				postMask[w] = _mm512_inserti32x4(wideR, upperLow, 2);
			else
				postMask[w] = _mm512_inserti32x4(wideR, lowerLow, 2);*/

			targetGateKey[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKeyR[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;
			targetPiBit[w] = currentGate->gs.yinput.pi + currentOffset;

			const uint8_t lsbitANDrsbit = (leftParentKey[15] & 0x01) & (rightParentKey[15] & 0x01);
			const uint8_t lsbitANDrsbit11 = (lsbitANDrsbit << 1) | lsbitANDrsbit;

			finalMask[w] = lsbitANDrsbit11 | (rpbit11 << 2) | (0x03 << 4) | (lsbitANDrsbit11 << 6);

			// this is a "tree" construction due to the high latency of the inserti instructions
			keys[w] = _mm512_castsi128_si512(lowerLow);
			__m256i upper = _mm256_castsi128_si256(lowerUpper);
			upper = _mm256_inserti128_si256(upper, upperUpper, 1);
			keys[w] = _mm512_inserti32x4(keys[w], upperLow, 1);
			keys[w] = _mm512_inserti64x4(keys[w], upper, 1);

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = counter;
			counter = _mm512_add_epi32(counter, COUNTER_DIFF);
		}

		for (size_t w = 0; w < width; ++w)
		{
			// this is the actual AES input
			data[w] = _mm512_xor_si512(data[w], keys[w]);
		}

		__m512i rcon = _mm512_set1_epi32(1);
		const __m512i shuffle_mask = _mm512_set1_epi32(0x0c0f0e0d);
		const __m512i rcon_multiplier = _mm512_set1_epi8(2);
		__m512i temp2[width], temp3[width];

		for (size_t r = 1; r < 10; ++r)
		{
			for (size_t w = 0; w < width; ++w)
			{
				temp2[w] = _mm512_shuffle_epi8(keys[w], shuffle_mask);
				temp2[w] = _mm512_aesenclast_epi128(temp2[w], rcon);
				// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
				temp3[w] = _mm512_bslli_epi128(keys[w], 0x4);
				keys[w] = _mm512_xor_si512(keys[w], temp3[w]);
				temp3[w] = _mm512_bslli_epi128(temp3[w], 0x4);
				keys[w] = _mm512_xor_si512(keys[w], temp3[w]);
				temp3[w] = _mm512_bslli_epi128(temp3[w], 0x4);
				keys[w] = _mm512_xor_si512(keys[w], temp3[w]);
				keys[w] = _mm512_xor_si512(keys[w], temp2[w]);

				data[w] = _mm512_aesenc_epi128(data[w], keys[w]);
			}

			rcon = _mm512_gf2p8mul_epi8(rcon, rcon_multiplier);
		}

		for (size_t w = 0; w < width; ++w)
		{
			temp2[w] = _mm512_shuffle_epi8(keys[w], shuffle_mask);
			temp2[w] = _mm512_aesenclast_epi128(temp2[w], rcon);
			// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
			temp3[w] = _mm512_bslli_epi128(keys[w], 0x4);
			keys[w] = _mm512_xor_si512(keys[w], temp3[w]);
			temp3[w] = _mm512_bslli_epi128(temp3[w], 0x4);
			keys[w] = _mm512_xor_si512(keys[w], temp3[w]);
			temp3[w] = _mm512_bslli_epi128(temp3[w], 0x4);
			keys[w] = _mm512_xor_si512(keys[w], temp3[w]);
			keys[w] = _mm512_xor_si512(keys[w], temp2[w]);

			data[w] = _mm512_aesenclast_epi128(data[w], keys[w]);
		}


		for (size_t w = 0; w < width; ++w)
		{
			// intent (shuffling into, in terms of 128-bit lanes):
			// 2
			// 0
			// 3
			// 0
			const __m512i shuffleKey = _mm512_set_epi64(
				1, 0,
				7, 6,
				1, 0,
				5, 4
			);
			__m512i shuffledCopy = _mm512_permutexvar_epi64(shuffleKey, data[w]);
			__m512i firstXor = _mm512_xor_si512(data[w], shuffledCopy);

			__m512i secondXor = _mm512_mask_xor_epi64(firstXor, finalMask[w], firstXor, postMask[w]);
			_mm_storeu_si128((__m128i*)gtptr, _mm512_extracti32x4_epi32(secondXor, 1));
			gtptr += 16;
			_mm_storeu_si128((__m128i*)gtptr, _mm512_extracti32x4_epi32(secondXor, 2));
			gtptr += 16;
			const __m128i nonRpbitOutKey = _mm512_extracti32x4_epi32(secondXor, 0);
			const __m128i rpbitRXor = _mm512_extracti32x4_epi32(firstXor, 2);
			const __m128i rpbitLXor = _mm512_extracti32x4_epi32(secondXor, 3);
			__m128i outKey = _mm_mask_xor_epi64(nonRpbitOutKey, rpbit[w], rpbitLXor, rpbitRXor);
			uint8_t outWireBit = _mm_extract_epi8(outKey, 15) & 0x01;
			*targetPiBit[w] ^= outWireBit;
			const uint8_t amplifiedOutWireBit = (outWireBit << 1) | outWireBit;
			outKey = _mm_mask_xor_epi64(outKey, amplifiedOutWireBit, outKey, R);

			_mm_storeu_si128((__m128i*)targetGateKey[w], outKey);
			_mm_storeu_si128((__m128i*)targetGateKeyR[w], _mm_xor_si128(outKey, R));
		}
	}
}