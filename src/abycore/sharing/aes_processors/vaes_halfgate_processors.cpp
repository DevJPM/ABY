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
	ProcessQueue(m_gateQueue, mainEvaluatingWidthVaes, numTablesInBatch, tableCounter, receivedTables);
}

void FixedKeyLTEvaluatingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthVaes>(wireCounter, 0, 0, numWiresInBatch, tableBuffer);
}

void FixedKeyLTEvaluatingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyLTGarblingVaesProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, mainGarblingWidthVaes, numTablesInBatch, tableCounter, tableBuffer);
}

template<size_t width>
inline void FixedKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
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
		__m128i leftTemp[width];
		__m128i rightTemp[width];
		__m128i finalTemp[width];
		// TODO: optimize using bigger vector loads potentially?
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

			leftTemp[w] = _mm_loadu_si128((__m128i*)leftParentKey);
			rightTemp[w] = _mm_loadu_si128((__m128i*)rightParentKey);

			targetGateKey[w] = currentGate->gs.yval + 16 * currentOffset;

			finalTemp[w] = _mm_setzero_si128();
			if (leftParentKey[15] & 0x01)
			{
				finalTemp[w] = _mm_loadu_si128((__m128i*)gtptr);
			}
			gtptr += 16;
			if (rightParentKey[15] & 0x01)
			{
				__m128i temp = _mm_loadu_si128((__m128i*)gtptr);
				finalTemp[w] = _mm_xor_si128(finalTemp[w], temp);
				finalTemp[w] = _mm_xor_si128(finalTemp[w], leftTemp[w]);
			}
			gtptr += 16;

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

		for (size_t w = 0; w < div_width; ++w)
		{
			//INSERT(rightKeys, rightTemp, 0);
			finalMask[w] = _mm512_castsi128_si512(finalTemp[4 * w]);
			INSERT(finalMask, finalTemp, 1);
			INSERT(finalMask, finalTemp, 2);
			INSERT(finalMask, finalTemp, 3);
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
			tempR = _mm512_shuffle_epi32(tempR, _MM_PERM_BADC); // 0x4E is 01 00 11 10 in binary which is exactly a 64-bit word lane swap
			const __m512i topExtractor = _mm512_set_epi64(
				~0, 0,
				~0, 0,
				~0, 0,
				~0, 0);
			__m512i topBit = _mm512_and_si512(tempR, topExtractor);
			leftKeys[w] = _mm512_xor_si512(tempL, topBit);

			tempL = _mm512_slli_epi64(rightKeys[w], 1);
			tempR = _mm512_srli_epi64(rightKeys[w], 63);
			tempR = _mm512_shuffle_epi32(tempR, _MM_PERM_BADC); // 0x4E is 01 00 11 10 in binary which is exactly a 64-bit word lane swap
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
			leftData[w] = _mm512_xor_si512(leftData[w], finalMask[w]);
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
			rpbit[w] = rightParent->gs.yinput.pi[currentOffset];

			const uint8_t rpbit11 = (rpbit[w] << 1) | rpbit[w];

			currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbit[w];

			const __m128i lowerLow = _mm_loadu_si128((__m128i*)leftParentKey);
			const __m128i upperLow = _mm_xor_si128(lowerLow, R);
			const __m128i lowerUpper = _mm_loadu_si128((__m128i*)rightParentKey);
			const __m128i upperUpper = _mm_xor_si128(lowerUpper, R);

			if (lpbit)
				postMask[w] = _mm512_inserti32x4(wideR, upperLow, 2);
			else
				postMask[w] = _mm512_inserti32x4(wideR, lowerLow, 2);

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
			// this assumes that we actually use the correct constant that sets the top bit
			// this is the left shift by 1 bit
			__m512i tempL = _mm512_slli_epi64(keys[w], 1);
			__m512i tempR = _mm512_srli_epi64(keys[w], 63);
			tempR = _mm512_shuffle_epi32(tempR, _MM_PERM_BADC); // 0x4E is 01 00 11 10 in binary which is exactly a 64-bit word lane swap, note that this instruction is within 128-bit lanes
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
			__m128i outKey;
			if (rpbit[w]) {
				__m128i rXor = _mm512_extracti32x4_epi32(firstXor, 2);
				outKey = _mm512_extracti32x4_epi32(secondXor, 3);
				outKey = _mm_xor_si128(outKey, rXor);
			}
			else {
				outKey = _mm512_extracti32x4_epi32(secondXor, 0);
			}
			uint8_t rBit = _mm_extract_epi8(R, 15) & 0x01;
			uint8_t outWireBit = _mm_extract_epi8(outKey, 15) & 0x01;
			if (outWireBit) {
				outKey = _mm_xor_si128(outKey, R);
				*targetPiBit[w] ^= rBit;
			}
				
			_mm_storeu_si128((__m128i*)targetGateKey[w], outKey);
			_mm_storeu_si128((__m128i*)targetGateKeyR[w], _mm_xor_si128(outKey,R));
		}
	}
}

void FixedKeyLTGarblingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthVaes>(wireCounter, numWiresInBatch, 0, 0, tableBuffer);
}

void FixedKeyLTGarblingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}
