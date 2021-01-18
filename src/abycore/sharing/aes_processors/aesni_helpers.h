#ifndef __AESNI_HELPERS_H__
#define __AESNI_HELPERS_H__

#include <wmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>

#include <iostream>
#include <iomanip>

template<size_t key_width, size_t data_per_key>
static inline __attribute__((always_inline)) void aesni_encrypt_variable_keys(__m128i keys[key_width], __m128i data[key_width * data_per_key]) {
	// this uses the fast AES key expansion (i.e. not using keygenassist) from
	// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
	// page 37

	const __m128i shuffle_mask =
		_mm_set_epi32(0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d);
	__m128i rcon;

	for (size_t w = 0; w < key_width; ++w) {
		for (size_t d = 0; d < data_per_key; ++d) {
			data[w * data_per_key + d] = _mm_xor_si128(data[w * data_per_key + d], keys[w]);
		}
	}

	rcon = _mm_set_epi32(1, 1, 1, 1);
	for (size_t r = 1; r <= 8; r++) {
		for (size_t w = 0; w < key_width; ++w)
		{
			__m128i temp2, temp3;
			temp2 = _mm_shuffle_epi8(keys[w], shuffle_mask);
			temp2 = _mm_aesenclast_si128(temp2, rcon);
			// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
			temp3 = _mm_slli_si128(keys[w], 0x4);
			keys[w] = _mm_xor_si128(keys[w], temp3);
			temp3 = _mm_slli_si128(temp3, 0x4);
			keys[w] = _mm_xor_si128(keys[w], temp3);
			temp3 = _mm_slli_si128(temp3, 0x4);
			keys[w] = _mm_xor_si128(keys[w], temp3);
			keys[w] = _mm_xor_si128(keys[w], temp2);

			for (size_t d = 0; d < data_per_key; ++d) {
				data[w * data_per_key + d] = _mm_aesenc_si128(data[w * data_per_key + d], keys[w]);
			}
		}
		rcon = _mm_slli_epi32(rcon, 1);
	}
	rcon = _mm_set_epi32(0x1b, 0x1b, 0x1b, 0x1b);

	for (size_t w = 0; w < key_width; ++w)
	{
		__m128i temp2, temp3;
		temp2 = _mm_shuffle_epi8(keys[w], shuffle_mask);
		temp2 = _mm_aesenclast_si128(temp2, rcon);
		// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
		temp3 = _mm_slli_si128(keys[w], 0x4);
		keys[w] = _mm_xor_si128(keys[w], temp3);
		temp3 = _mm_slli_si128(temp3, 0x4);
		keys[w] = _mm_xor_si128(keys[w], temp3);
		temp3 = _mm_slli_si128(temp3, 0x4);
		keys[w] = _mm_xor_si128(keys[w], temp3);
		keys[w] = _mm_xor_si128(keys[w], temp2);

		for (size_t d = 0; d < data_per_key; ++d) {
			data[w * data_per_key + d] = _mm_aesenc_si128(data[w * data_per_key + d], keys[w]);
		}
	}
	rcon = _mm_slli_epi32(rcon, 1);

	for (size_t w = 0; w < key_width; ++w)
	{
		__m128i temp2, temp3;
		temp2 = _mm_shuffle_epi8(keys[w], shuffle_mask);
		temp2 = _mm_aesenclast_si128(temp2, rcon);
		temp3 = _mm_slli_si128(keys[w], 0x4);
		keys[w] = _mm_xor_si128(keys[w], temp3);
		temp3 = _mm_slli_si128(temp3, 0x4);
		keys[w] = _mm_xor_si128(keys[w], temp3);
		temp3 = _mm_slli_si128(temp3, 0x4);
		keys[w] = _mm_xor_si128(keys[w], temp3);
		keys[w] = _mm_xor_si128(keys[w], temp2);

		for (size_t d = 0; d < data_per_key; ++d) {
			data[w * data_per_key + d] = _mm_aesenclast_si128(data[w * data_per_key + d], keys[w]);
		}
	}
}

static inline void PrintKey(__m128i data) {
	uint8_t key[16];
	_mm_storeu_si128((__m128i*)key, data);

	for (uint32_t i = 0; i < 16; i++) {
		std::cout << std::setw(2) << std::setfill('0') << (std::hex) << (uint32_t)key[i];
	}
	std::cout << (std::dec);
}

#endif
