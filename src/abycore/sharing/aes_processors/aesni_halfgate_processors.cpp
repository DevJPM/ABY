#include "aesni_halfgate_processors.h"
#include "../yaosharing.h"

#include <wmmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <iostream>
#include <iomanip>
#include <algorithm>

// in number of tables
constexpr size_t mainGarblingWidthNI = 2;
constexpr size_t mainEvaluatingWidthNI = 4;

static void PrintKey(__m128i data) {
	uint8_t key[16];
	_mm_storeu_si128((__m128i*)key, data);

	for (uint32_t i = 0; i < 16; i++) {
		std::cout << std::setw(2) << std::setfill('0') << (std::hex) << (uint32_t)key[i];
	}
	std::cout << (std::dec);
}

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void FixedKeyLTGarblingAesniProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[4 * width];
	__m128i keys[4 * width];
	__m128i aes_keys[11];
	__m128i rtable[width];
	uint8_t* targetGateKey[width];
	uint8_t* targetGateKeyR[width];
	uint8_t lsbitANDrsbit[width];
	uint8_t rpbit[width];
	uint8_t* targetPiBit[width];

	for (size_t i = 0; i < 11; ++i)
	{
		aes_keys[i] = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
	}

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
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

			currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbit[w];
			targetPiBit[w] = currentGate->gs.yinput.pi + currentOffset;

			keys[4 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey);
			keys[4 * w + 1] = _mm_xor_si128(keys[4 * w + 0], R);
			keys[4 * w + 2] = _mm_loadu_si128((__m128i*)rightParentKey);
			keys[4 * w + 3] = _mm_xor_si128(keys[4 * w + 2], R);

			targetGateKey[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKeyR[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;

			lsbitANDrsbit[w] = (leftParentKey[15] & 0x01) & (rightParentKey[15] & 0x01);

			if (lpbit)
				rtable[w] = keys[4 * w + 1];
			else
				rtable[w] = keys[4 * w + 0];

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[4 * w + 0] = counter;
			data[4 * w + 1] = counter;
			data[4 * w + 2] = _mm_add_epi32(counter, ONE);
			data[4 * w + 3] = data[4 * w + 2];
			counter = _mm_add_epi32(counter, TWO);
		}

		for (size_t w = 0; w < 4 * width; ++w)
		{
			// this is the 128-bit leftshift code from https://stackoverflow.com/a/34482688/4733946
			// as requested by User0 https://stackoverflow.com/users/5720018/user0
			// and given by Peter Cordes https://stackoverflow.com/users/224132/peter-cordes

			__m128i carry = _mm_bslli_si128(keys[w], 8);
			carry = _mm_srli_epi64(carry, 63);
			keys[w] = _mm_slli_epi64(keys[w], 1);
			keys[w] = _mm_or_si128(keys[w], carry);

			// this is the actual AES input
			data[w] = _mm_xor_si128(data[w], keys[w]);

			keys[w] = data[w]; // keep as a backup for post-whitening
			data[w] = _mm_xor_si128(data[w], aes_keys[0]);
		}
			
		for (size_t r = 1; r < 10; ++r)
		{
			for (size_t w = 0; w < 4 * width; ++w)
			{
				data[w] = _mm_aesenc_si128(data[w], aes_keys[r]);
			}
		}

		for (size_t w = 0; w < 4 * width; ++w)
		{
			data[w] = _mm_aesenclast_si128(data[w], aes_keys[10]);
			data[w] = _mm_xor_si128(data[w], keys[w]);
		}
			

		for (size_t w = 0; w < width; ++w)
		{
			__m128i ltable = _mm_xor_si128(data[4 * w + 0], data[4 * w + 1]);
			if (rpbit[w])
				ltable = _mm_xor_si128(ltable, R);
			_mm_storeu_si128((__m128i*)gtptr, ltable);
			gtptr += 16;
			const __m128i rXor = _mm_xor_si128(data[4 * w + 2], data[4 * w + 3]);
			rtable[w] = _mm_xor_si128(rtable[w], rXor);
			_mm_storeu_si128((__m128i*)gtptr, rtable[w]);
			gtptr += 16;

			__m128i outKey;
			if (rpbit[w])
				outKey = _mm_xor_si128(data[4 * w + 0], data[4 * w + 3]);
			else
				outKey = _mm_xor_si128(data[4 * w + 0], data[4 * w + 2]);
			if (lsbitANDrsbit[w])
				outKey = _mm_xor_si128(outKey, R);
			if (rpbit[w])
				outKey = _mm_xor_si128(outKey, rXor);
			
			uint8_t rBit = _mm_extract_epi8(R, 15) & 0x01;
			uint8_t outWireBit = _mm_extract_epi8(outKey, 15) & 0x01;
			if (outWireBit)
			{
				outKey = _mm_xor_si128(outKey, R);
				*targetPiBit[w] ^= rBit;
			}
				
			_mm_storeu_si128((__m128i*)targetGateKey[w], outKey);
			_mm_storeu_si128((__m128i*)targetGateKeyR[w], _mm_xor_si128(outKey,R));
		}
	}
}

void FixedKeyLTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t FixedKeyLTGarblingAesniProcessor::vectorWidth() const
{
	return mainGarblingWidthNI;
}

void FixedKeyLTGarblingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthNI>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void FixedKeyLTGarblingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void InputKeyLTGarblingAesniProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[4 * width];
	__m128i parentKeys[4 * width];

	__m128i rtable[width];
	uint8_t* targetGateKey[width];
	uint8_t* targetGateKeyR[width];
	uint8_t lsbitANDrsbit[width];
	uint8_t rpbit[width];
	uint8_t* targetPiBit[width];

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);

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

			currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbit[w];
			targetPiBit[w] = currentGate->gs.yinput.pi + currentOffset;

			parentKeys[4 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey);
			parentKeys[4 * w + 1] = _mm_xor_si128(parentKeys[4 * w + 0], R);
			parentKeys[4 * w + 2] = _mm_loadu_si128((__m128i*)rightParentKey);
			parentKeys[4 * w + 3] = _mm_xor_si128(parentKeys[4 * w + 2], R);

			targetGateKey[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKeyR[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;

			lsbitANDrsbit[w] = (leftParentKey[15] & 0x01) & (rightParentKey[15] & 0x01);

			if (lpbit)
				rtable[w] = parentKeys[4 * w + 1];
			else
				rtable[w] = parentKeys[4 * w + 0];

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[4 * w + 0] = counter;
			data[4 * w + 1] = counter;
			data[4 * w + 2] = _mm_add_epi32(counter, ONE);
			data[4 * w + 3] = data[4 * w + 2];
			counter = _mm_add_epi32(counter, TWO);
		}

		for (size_t w = 0; w < 4 * width; ++w)
		{
			data[w] = _mm_xor_si128(data[w], parentKeys[w]);
		}

		// this uses the fast AES key expansion (i.e. not using keygenassist) from
		// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
		// page 37

		__m128i temp2[4*width], temp3[4*width];
		const __m128i shuffle_mask =
			_mm_set_epi32(0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d);
		__m128i rcon;

		rcon = _mm_set_epi32(1, 1, 1, 1);
		for (int r = 1; r <= 8; r++) {
			for (size_t w = 0; w < 4 * width; ++w)
			{
				temp2[w] = _mm_shuffle_epi8(parentKeys[w], shuffle_mask);
				temp2[w] = _mm_aesenclast_si128(temp2[w], rcon);
				// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
				temp3[w] = _mm_slli_si128(parentKeys[w], 0x4);
				parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
				temp3[w] = _mm_slli_si128(temp3[w], 0x4);
				parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
				temp3[w] = _mm_slli_si128(temp3[w], 0x4);
				parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
				parentKeys[w] = _mm_xor_si128(parentKeys[w], temp2[w]);

				data[w] = _mm_aesenc_si128(data[w], parentKeys[w]);
			}	
			rcon = _mm_slli_epi32(rcon, 1);
		}
		rcon = _mm_set_epi32(0x1b, 0x1b, 0x1b, 0x1b);

		for (size_t w = 0; w < 4 * width; ++w)
		{
			temp2[w] = _mm_shuffle_epi8(parentKeys[w], shuffle_mask);
			temp2[w] = _mm_aesenclast_si128(temp2[w], rcon);
			// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
			temp3[w] = _mm_slli_si128(parentKeys[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			temp3[w] = _mm_slli_si128(temp3[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			temp3[w] = _mm_slli_si128(temp3[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp2[w]);
			data[w] = _mm_aesenc_si128(data[w], parentKeys[w]);
		}
		rcon = _mm_slli_epi32(rcon, 1);

		for (size_t w = 0; w < 4 * width; ++w)
		{
			temp2[w] = _mm_shuffle_epi8(parentKeys[w], shuffle_mask);
			temp2[w] = _mm_aesenclast_si128(temp2[w], rcon);
			temp3[w] = _mm_slli_si128(parentKeys[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			temp3[w] = _mm_slli_si128(temp3[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			temp3[w] = _mm_slli_si128(temp3[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp2[w]);
			data[w] = _mm_aesenclast_si128(data[w], parentKeys[w]);
		}

		for (size_t w = 0; w < 4 * width; ++w)
		{
			__m128i ltable = _mm_xor_si128(data[4 * w + 0], data[4 * w + 1]);
			if (rpbit[w])
				ltable = _mm_xor_si128(ltable, R);
			_mm_storeu_si128((__m128i*)gtptr, ltable);
			gtptr += 16;
			const __m128i rXor = _mm_xor_si128(data[4 * w + 2], data[4 * w + 3]);
			rtable[w] = _mm_xor_si128(rtable[w], rXor);
			_mm_storeu_si128((__m128i*)gtptr, rtable[w]);
			gtptr += 16;

			__m128i outKey;
			if (rpbit[w])
				outKey = _mm_xor_si128(data[4 * w + 0], data[4 * w + 3]);
			else
				outKey = _mm_xor_si128(data[4 * w + 0], data[4 * w + 2]);
			if (lsbitANDrsbit[w])
				outKey = _mm_xor_si128(outKey, R);
			if (rpbit[w])
				outKey = _mm_xor_si128(outKey, rXor);
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

void InputKeyLTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t InputKeyLTGarblingAesniProcessor::vectorWidth() const
{
	return mainGarblingWidthNI;
}

void InputKeyLTGarblingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthNI>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void InputKeyLTGarblingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

static void expandAESKey(__m128i userkey, uint8_t* alignedStoragePointer)
{
	// this uses the fast AES key expansion (i.e. not using keygenassist) from
	// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
	// page 37

	__m128i temp1, temp2, temp3;
	__m128i shuffle_mask =
		_mm_set_epi32(0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d);
	__m128i rcon;
	temp1 = userkey;
	rcon = _mm_set_epi32(1, 1, 1, 1);
	_mm_store_si128((__m128i*)(alignedStoragePointer + 0 * 16), temp1);
	for (int i = 1; i <= 8; i++) {
		temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
		temp2 = _mm_aesenclast_si128(temp2, rcon);
		rcon = _mm_slli_epi32(rcon, 1);
		temp3 = _mm_slli_si128(temp1, 0x4);
		temp1 = _mm_xor_si128(temp1, temp3);
		temp3 = _mm_slli_si128(temp3, 0x4);
		temp1 = _mm_xor_si128(temp1, temp3);
		temp3 = _mm_slli_si128(temp3, 0x4);
		temp1 = _mm_xor_si128(temp1, temp3);
		temp1 = _mm_xor_si128(temp1, temp2);
		_mm_store_si128((__m128i*)(alignedStoragePointer + i * 16), temp1);
	}
	rcon = _mm_set_epi32(0x1b, 0x1b, 0x1b, 0x1b);
	temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
	temp2 = _mm_aesenclast_si128(temp2, rcon);
	rcon = _mm_slli_epi32(rcon, 1);
	temp3 = _mm_slli_si128(temp1, 0x4);
	temp1 = _mm_xor_si128(temp1, temp3);
	temp3 = _mm_slli_si128(temp3, 0x4);
	temp1 = _mm_xor_si128(temp1, temp3);
	temp3 = _mm_slli_si128(temp3, 0x4);
	temp1 = _mm_xor_si128(temp1, temp3);
	temp1 = _mm_xor_si128(temp1, temp2);
	_mm_store_si128((__m128i*)(alignedStoragePointer + 9 * 16), temp1);
	temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
	temp2 = _mm_aesenclast_si128(temp2, rcon);
	temp3 = _mm_slli_si128(temp1, 0x4);
	temp1 = _mm_xor_si128(temp1, temp3);
	temp3 = _mm_slli_si128(temp3, 0x4);
	temp1 = _mm_xor_si128(temp1, temp3);
	temp3 = _mm_slli_si128(temp3, 0x4);
	temp1 = _mm_xor_si128(temp1, temp3);
	temp1 = _mm_xor_si128(temp1, temp2);
	_mm_store_si128((__m128i*)(alignedStoragePointer + 10 * 16), temp1);
}


template<size_t width>
inline void FixedKeyLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[2 * width];
	__m128i keys[2 * width];
	__m128i finalMask[width];
	uint8_t* targetGateKey[width];
	__m128i aes_keys[11];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;

	for (size_t i = 0; i < 11; ++i)
	{
		aes_keys[i] = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
	}

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

			keys[2 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey);
			keys[2 * w + 1] = _mm_loadu_si128((__m128i*)rightParentKey);

			targetGateKey[w] = currentGate->gs.yval + 16 * currentOffset;

			finalMask[w] = _mm_setzero_si128();
			if (leftParentKey[15] & 0x01)
			{
				finalMask[w] = _mm_loadu_si128((__m128i*)gtptr);
			}
			gtptr += 16;
			if (rightParentKey[15] & 0x01)
			{
				__m128i temp = _mm_loadu_si128((__m128i*)gtptr);
				finalMask[w] = _mm_xor_si128(finalMask[w], temp);
				finalMask[w] = _mm_xor_si128(finalMask[w],keys[2 * w + 0]);
			}
			gtptr += 16;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[2 * w + 0] = counter;
			data[2 * w + 1] = _mm_add_epi32(counter, ONE);
			counter = _mm_add_epi32(counter, TWO);
		}

		for (size_t w = 0; w < 2 * width; ++w)
		{
			// this is the 128-bit leftshift code from https://stackoverflow.com/a/34482688/4733946
			// as requested by User0 https://stackoverflow.com/users/5720018/user0
			// and given by Peter Cordes https://stackoverflow.com/users/224132/peter-cordes

			__m128i carry = _mm_bslli_si128(keys[w], 8);
			carry = _mm_srli_epi64(carry, 63);
			keys[w] = _mm_slli_epi64(keys[w], 1);
			keys[w] = _mm_or_si128(keys[w], carry);

			// this is the actual AES input
			data[w] = _mm_xor_si128(data[w], keys[w]);

			keys[w] = data[w]; // keep as a backup for post-whitening
			data[w] = _mm_xor_si128(data[w], aes_keys[0]);
		}

		for (size_t r = 1; r < 10; ++r)
		{
			for (size_t w = 0; w < 2 * width; ++w)
			{
				data[w] = _mm_aesenc_si128(data[w], aes_keys[r]);
			}
		}

		for (size_t w = 0; w < 2 * width; ++w)
		{
			data[w] = _mm_aesenclast_si128(data[w], aes_keys[10]);
			data[w] = _mm_xor_si128(data[w], keys[w]);
		}


		for (size_t w = 0; w < width; ++w)
		{
			__m128i temp = _mm_xor_si128(data[2 * w + 0], data[2 * w + 1]);
			temp = _mm_xor_si128(temp, finalMask[w]);
			_mm_storeu_si128((__m128i*)(targetGateKey[w]), temp);
		}	
	}
}


template<size_t width>
inline void InputKeyLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[2 * width];
	__m128i parentKeys[2 * width];
	__m128i finalMask[width];
	uint8_t* targetGateKey[width];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;


	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

			parentKeys[2 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey);
			parentKeys[2 * w + 1] = _mm_loadu_si128((__m128i*)rightParentKey);

			targetGateKey[w] = currentGate->gs.yval + 16 * currentOffset;

			finalMask[w] = _mm_setzero_si128();
			if (leftParentKey[15] & 0x01)
			{
				finalMask[w] = _mm_loadu_si128((__m128i*)gtptr);
			}
			gtptr += 16;
			if (rightParentKey[15] & 0x01)
			{
				__m128i temp = _mm_loadu_si128((__m128i*)gtptr);
				finalMask[w] = _mm_xor_si128(finalMask[w], temp);
				finalMask[w] = _mm_xor_si128(finalMask[w], parentKeys[2 * w + 0]);
			}
			gtptr += 16;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[2 * w + 0] = counter;
			data[2 * w + 1] = _mm_add_epi32(counter, ONE);
			counter = _mm_add_epi32(counter, TWO);
		}

		for (size_t w = 0; w < 2 * width; ++w)
		{
			data[w] = _mm_xor_si128(data[w], parentKeys[w]);
		}

		// this uses the fast AES key expansion (i.e. not using keygenassist) from
		// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
		// page 37

		__m128i temp2[4 * width], temp3[4 * width];
		const __m128i shuffle_mask =
			_mm_set_epi32(0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d);
		__m128i rcon;

		rcon = _mm_set_epi32(1, 1, 1, 1);
		for (int r = 1; r <= 8; r++) {
			for (size_t w = 0; w < 2 * width; ++w)
			{
				temp2[w] = _mm_shuffle_epi8(parentKeys[w], shuffle_mask);
				temp2[w] = _mm_aesenclast_si128(temp2[w], rcon);
				// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
				temp3[w] = _mm_slli_si128(parentKeys[w], 0x4);
				parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
				temp3[w] = _mm_slli_si128(temp3[w], 0x4);
				parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
				temp3[w] = _mm_slli_si128(temp3[w], 0x4);
				parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
				parentKeys[w] = _mm_xor_si128(parentKeys[w], temp2[w]);

				data[w] = _mm_aesenc_si128(data[w], parentKeys[w]);
			}
			rcon = _mm_slli_epi32(rcon, 1);
		}
		rcon = _mm_set_epi32(0x1b, 0x1b, 0x1b, 0x1b);

		for (size_t w = 0; w < 2 * width; ++w)
		{
			temp2[w] = _mm_shuffle_epi8(parentKeys[w], shuffle_mask);
			temp2[w] = _mm_aesenclast_si128(temp2[w], rcon);
			// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
			temp3[w] = _mm_slli_si128(parentKeys[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			temp3[w] = _mm_slli_si128(temp3[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			temp3[w] = _mm_slli_si128(temp3[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp2[w]);
			data[w] = _mm_aesenc_si128(data[w], parentKeys[w]);
		}
		rcon = _mm_slli_epi32(rcon, 1);

		for (size_t w = 0; w < 2 * width; ++w)
		{
			temp2[w] = _mm_shuffle_epi8(parentKeys[w], shuffle_mask);
			temp2[w] = _mm_aesenclast_si128(temp2[w], rcon);
			temp3[w] = _mm_slli_si128(parentKeys[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			temp3[w] = _mm_slli_si128(temp3[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			temp3[w] = _mm_slli_si128(temp3[w], 0x4);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp3[w]);
			parentKeys[w] = _mm_xor_si128(parentKeys[w], temp2[w]);
			data[w] = _mm_aesenclast_si128(data[w], parentKeys[w]);
		}


		for (size_t w = 0; w < width; ++w)
		{
			__m128i temp = _mm_xor_si128(data[2 * w + 0], data[2 * w + 1]);
			temp = _mm_xor_si128(temp, finalMask[w]);
			_mm_storeu_si128((__m128i*)(targetGateKey[w]), temp);
		}
	}
}

void FixedKeyLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t FixedKeyLTEvaluatingAesniProcessor::vectorWidth() const
{
	return mainEvaluatingWidthNI;
}

void FixedKeyLTEvaluatingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthNI>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyLTEvaluatingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void InputKeyLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t InputKeyLTEvaluatingAesniProcessor::vectorWidth() const
{
	return mainEvaluatingWidthNI;
}

void InputKeyLTEvaluatingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthNI>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void InputKeyLTEvaluatingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyProvider::expandAESKey(const uint8_t* userkey)
{
	if (userkey) {
		__m128i key = _mm_loadu_si128((__m128i*)userkey);
		::expandAESKey(key, m_expandedStaticAESKey.get());
	}
	else {
		// note that order is most significant to least significant byte for this intrinsic
		const __m128i key = _mm_set_epi8(0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00);
		::expandAESKey(key, m_expandedStaticAESKey.get());
	}
}
