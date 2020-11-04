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

static void PrintKey(__m512i data) {
	uint8_t key[64];
	_mm512_storeu_si512((__m512i*)key, data);

	for (int j = 0; j < 64; j += 16)
	{
		for (uint32_t i = 0; i < 16; i++) {
			std::cout << std::setw(2) << std::setfill('0') << (std::hex) << (uint32_t)key[i+j];
		}
		std::cout << std::endl;
	}
	
	std::cout << (std::dec);
}

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void FixedKeyLTGarblingAesniProcessor::fillAESBufferAND(size_t baseOffset,uint32_t tableCounter, size_t numTablesInBatch,size_t bufferOffset)
{
	std::cout << "garbling AES-NI" << std::endl;

	assert(bufferOffset + numTablesInBatch * 4 * 16 <= m_bufferSize);

	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[4 * width];
	__m128i keys[4 * width];
	__m128i aes_keys[11];

	for (size_t i = 0; i < 11; ++i)
	{
		aes_keys[i] = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
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

			keys[4 * w + 0] = _mm_loadu_si128((__m128i*)(leftParent->gs.yinput.outKey + 16 * simdOffset));
			keys[4 * w + 1] = _mm_xor_si128(keys[4 * w + 0], R);
			keys[4 * w + 2] = _mm_loadu_si128((__m128i*)(rightParent->gs.yinput.outKey + 16 * simdOffset));
			keys[4 * w + 3] = _mm_xor_si128(keys[4 * w + 2], R);
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
			// this assumes that we actually use the correct constant that sets the top bit
			// this is the left shift by 1 bit
			__m128i tempL = _mm_slli_epi64(keys[w], 1);
			__m128i tempR = _mm_srli_epi64(keys[w], 63);
			tempR = _mm_shuffle_epi32(tempR,0x4E); // 0x4E is 01 00 11 10 in binary which is exactly a 64-bit word lane swap
			__m128i topExtractor = _mm_set_epi64x(~0, 0);
			__m128i topBit = _mm_and_si128(tempR,topExtractor);
			keys[w] = _mm_xor_si128(tempL, topBit);

			//parentKeys[w] = _mm_slli_si128(parentKeys[w], 1); // this does BYTE shift not BIT shifts!1!

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
			

		for (size_t w = 0; w < 4 * width; ++w)
			_mm_store_si128((__m128i*)(m_aesBuffer + bufferOffset + (i * 4 + w) * 16), data[w]);
	}
}

void FixedKeyLTGarblingAesniProcessor::fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch)
{
	const size_t leftovers = numTablesInBatch % mainGarblingWidthNI;
	const size_t mainBulkSize = numTablesInBatch - leftovers;

	fillAESBufferAND<mainGarblingWidthNI>(baseOffset, tableCounter, mainBulkSize,0);

	if (leftovers > 0)
	{
		fillAESBufferAND<1>(baseOffset + mainBulkSize, tableCounter + mainBulkSize, leftovers, mainBulkSize * 16*4); // 16 bytes per ciphertext, 4 per table
	}
}

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void InputKeyLTGarblingAesniProcessor::fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch, size_t bufferOffset)
{
	assert(bufferOffset + numTablesInBatch * 4 * 16 <= m_bufferSize);

	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[4 * width];
	__m128i parentKeys[4 * width];

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

			parentKeys[4 * w + 0] = _mm_loadu_si128((__m128i*)(leftParent->gs.yinput.outKey + 16 * simdOffset));
			parentKeys[4 * w + 1] = _mm_xor_si128(parentKeys[4 * w + 0], R);
			parentKeys[4 * w + 2] = _mm_loadu_si128((__m128i*)(rightParent->gs.yinput.outKey + 16 * simdOffset));
			parentKeys[4 * w + 3] = _mm_xor_si128(parentKeys[4 * w + 2], R);
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
			_mm_store_si128((__m128i*)(m_aesBuffer + bufferOffset + (i * 4 + w) * 16), data[w]);
	}
}

void InputKeyLTGarblingAesniProcessor::fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch)
{
	const size_t leftovers = numTablesInBatch % mainGarblingWidthNI;
	const size_t mainBulkSize = numTablesInBatch - leftovers;

	fillAESBufferAND<mainGarblingWidthNI>(baseOffset, tableCounter, mainBulkSize, 0);

	if (leftovers > 0)
	{
		fillAESBufferAND<1>(baseOffset + mainBulkSize, tableCounter + mainBulkSize, leftovers, mainBulkSize * 16 * 4); // 16 bytes per ciphertext, 4 per table
	}
}

void FixedKeyProvider::expandAESKey()
{
	// this uses the fast AES key expansion (i.e. not using keygenassist) from
	// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
	// page 37

	uint8_t* alignedStoragePointer = m_expandedStaticAESKey.get();

	__m128i temp1, temp2, temp3;
	__m128i shuffle_mask =
		_mm_set_epi32(0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d);
	__m128i rcon;
	// note that order is most significant to least significant byte for this intrinsic
	const __m128i userkey = _mm_set_epi8(0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00);
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
inline void FixedKeyLTEvaluatingAesniProcessor::computeAESPreOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[2 * width];
	__m128i keys[2 * width];
	uint8_t* targetGateKey[width];
	__m128i aes_keys[11];

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

			keys[2 * w + 0] = _mm_loadu_si128((__m128i*)(leftParent->gs.yval + 16 * currentOffset));
			keys[2 * w + 1] = _mm_loadu_si128((__m128i*)(rightParent->gs.yval + 16 * currentOffset));

			targetGateKey[w] = currentGate->gs.yval + 16 * currentOffset;

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
			// this assumes that we actually use the correct constant that sets the top bit
			// this is the left shift by 1 bit
			__m128i tempL = _mm_slli_epi64(keys[w], 1);
			__m128i tempR = _mm_srli_epi64(keys[w], 63);
			tempR = _mm_shuffle_epi32(tempR, 0x4E); // 0x4E is 01 00 11 10 in binary which is exactly a 64-bit word lane swap
			__m128i topExtractor = _mm_set_epi64x(~0, 0);
			__m128i topBit = _mm_and_si128(tempR, topExtractor);
			keys[w] = _mm_xor_si128(tempL, topBit);

			//parentKeys[w] = _mm_slli_si128(parentKeys[w], 1); // this does BYTE shift not BIT shifts!1!

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
			__m128 temp = _mm_xor_si128(data[2 * w + 0], data[2 * w + 1]);
			_mm_storeu_si128((__m128i*)(targetGateKey[w]), temp);
		}	
	}
}


template<size_t width>
inline void InputKeyLTEvaluatingAesniProcessor::computeAESPreOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[2 * width];
	__m128i parentKeys[2 * width];
	uint8_t* targetGateKey[width];


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

			parentKeys[2 * w + 0] = _mm_loadu_si128((__m128i*)(leftParent->gs.yval + 16 * currentOffset));
			parentKeys[2 * w + 1] = _mm_loadu_si128((__m128i*)(rightParent->gs.yval + 16 * currentOffset));

			targetGateKey[w] = currentGate->gs.yval + 16 * currentOffset;

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
			__m128 temp = _mm_xor_si128(data[2 * w + 0], data[2 * w + 1]);
			_mm_storeu_si128((__m128i*)(targetGateKey[w]), temp);
		}
	}
}


void FixedKeyLTEvaluatingAesniProcessor::computeAESPreOutKeys(uint32_t tableCounter, size_t numTablesInBatch)
{
	if (m_gateQueue.size() == 0)
		return;

	const size_t leftovers = numTablesInBatch % mainEvaluatingWidthNI;
	const size_t mainBulkSize = numTablesInBatch - leftovers;

	computeAESPreOutKeys<mainEvaluatingWidthNI>(tableCounter,0,0, mainBulkSize);

	size_t numTablesLeft = 0;
	size_t ridx;

	for (ridx = m_gateQueue.size()-1; ridx >= 0; --ridx)
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



void InputKeyLTEvaluatingAesniProcessor::computeAESPreOutKeys(uint32_t tableCounter, size_t numTablesInBatch)
{
	if (m_gateQueue.size() == 0)
		return;

	const size_t leftovers = numTablesInBatch % mainEvaluatingWidthNI;
	const size_t mainBulkSize = numTablesInBatch - leftovers;

	computeAESPreOutKeys<mainEvaluatingWidthNI>(tableCounter, 0, 0, mainBulkSize);

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
