#include "prf_server.h"
#include "../../aby/abysetup.h"
#include "../aes_processors/aesni_prf_processors.h"
#include <wmmintrin.h>

void PRFServerSharing::InitServer()
{
	piCounter = 0;

	m_xorAESProcessor = std::make_unique<PRFXorLTGarblingAesniProcessor>(getXorQueue(), m_vGates, m_vXorIds);
	m_andAESProcessor = std::make_unique<PRFAndLTGarblingAesniProcessor>(getAndQueue(), m_vGates, m_vAndIds);

	CBitVector piKey;
	piKey.Create(128, m_cCrypto);
	m_andPiBitProvider = std::make_unique<FixedKeyProvider>(piKey.GetArr());
}

void PRFServerSharing::choosePi(GATE* gate)
{
	uint8_t buffer[64];
	__m128i data[4];
	uint8_t* piPtr = gate->gs.yinput.pi;
	__m128i mask = _mm_set1_epi8(0x01);

	for (int64_t numBytesLeft = gate->nvals; numBytesLeft >= 0; numBytesLeft -= 64, piPtr+=64) {
		const uint8_t* keyptr = m_andPiBitProvider->getExpandedStaticKey();

		for (int w = 0; w < 4; ++w) {
			data[w] = _mm_set_epi64x(0, piCounter);
			piCounter++;
			data[w] = _mm_xor_si128(data[w], _mm_load_si128((__m128i*)keyptr));
		}
		keyptr += 16;

		for (int i = 1; i < 10; ++i) {
			for (int w = 0; w < 4; ++w) {
				data[w] = _mm_aesenc_si128(data[w], _mm_load_si128((__m128i*)keyptr));
			}
			keyptr += 16;
		}

		for (int w = 0; w < 4; ++w) {
			data[w] = _mm_aesenclast_si128(data[w], _mm_load_si128((__m128i*)keyptr));
			data[w] = _mm_and_si128(mask, data[w]);
			_mm_storeu_si128((__m128i*)(buffer + w * 16), data[w]);
		}

		memcpy(piPtr, buffer, std::min(numBytesLeft, int64_t(64)));
	}
}

bool PRFServerSharing::evaluateConstantGate(GATE* gate)
{
	//assign 0 and 1 gates
	UGATE_T constval = gate->gs.constval;
	InstantiateGate(gate);
	memset(gate->gs.yinput.outKey[0], 0, m_nSecParamBytes * gate->nvals);
	memset(gate->gs.yinput.outKey[1], 0, m_nSecParamBytes * gate->nvals);
	for (uint32_t i = 0; i < gate->nvals; ++i) {
		if (constval == 1L) {
			gate->gs.yinput.pi[i] = 1;
		}
		else {
			gate->gs.yinput.pi[i] = 0;
		}
	}

	return true;
}

void PRFServerSharing::createOppositeInputKeys(CBitVector& oppositeInputKeys, CBitVector& regularInputKeys, size_t numKeys)
{
	const size_t numBytes = numKeys * m_nSecParamBytes;

	//BYTE* buffer = (BYTE*)malloc(numBytes);
	//oppositeInputKeys.AttachBuf(buffer, numBytes);
	oppositeInputKeys.Create(numBytes * 8, m_cCrypto);
	for (uint32_t i = 0; i < numKeys; i++) {
		oppositeInputKeys.ORByte((i + 1) * m_nSecParamBytes - 1, 0x01);
		assert(oppositeInputKeys.GetArr()[(i + 1) * m_nSecParamBytes - 1] & 0x01);
		assert(!(regularInputKeys.GetArr()[(i + 1) * m_nSecParamBytes - 1] & 0x01));
	}
}

void PRFServerSharing::evaluateDeferredXORGates(size_t numWires)
{
	// the buffers are needed for the batch processing
	for (auto* currentGate : getXorQueue())
		InstantiateGate(currentGate);

	// we call into another class here as this allows us to exploit dynamic dispatch
	// to switch between AES256, AES128, AES-NI and VAES as needed based on a ctor parameter
	// This essentially does the wire encrypting previously done just-in-time
	m_xorAESProcessor->computeAESOutKeys(m_nXorGateTableCtr, numWires, m_vXorGateTable.GetArr());

	m_nXorGateTableCtr += numWires;

	for (auto* currentGate : getXorQueue())
	{
		uint32_t idleft = currentGate->ingates.inputs.twin.left; //gate->gs.ginput.left;
		uint32_t idright = currentGate->ingates.inputs.twin.right; //gate->gs.ginput.right;

		UsedGate(idleft);
		UsedGate(idright);
	}
}

void PRFServerSharing::evaluateDeferredANDGates(ABYSetup* setup, size_t numWires)
{
	// the buffers are needed for the batch processing
	// as are the random choices for the permutation bits
	for (auto* currentGate : getAndQueue()) {
		InstantiateGate(currentGate);
		choosePi(currentGate);
	}
		

	// we call into another class here as this allows us to exploit dynamic dispatch
	// to switch between AES256, AES128, AES-NI and VAES as needed based on a ctor parameter
	// This essentially does the wire encrypting previously done just-in-time
	m_andAESProcessor->computeAESOutKeys(m_nAndGateTableCtr, numWires, m_vAndGateTable.GetArr());

	m_nAndGateTableCtr += numWires;

	for (auto* currentGate : getAndQueue())
	{
		uint32_t idleft = currentGate->ingates.inputs.twin.left; //gate->gs.ginput.left;
		uint32_t idright = currentGate->ingates.inputs.twin.right; //gate->gs.ginput.right;

		UsedGate(idleft);
		UsedGate(idright);
	}

	if ((m_nAndGateTableCtr - m_nGarbledTableSndCtr) >= GARBLED_TABLE_WINDOW)
	{
		setup->AddSendTask(m_vAndGateTable.GetArr() + m_nGarbledTableSndCtr * m_nSecParamBytes * KEYS_PER_GATE_IN_TABLE,
			(m_nAndGateTableCtr - m_nGarbledTableSndCtr) * m_nSecParamBytes * KEYS_PER_GATE_IN_TABLE);
		m_nGarbledTableSndCtr = m_nAndGateTableCtr;
	}
}

bool PRFServerSharing::evaluateUNIVGate(GATE* gate)
{
	uint32_t idleft = gate->ingates.inputs.twin.left;
	uint32_t idright = gate->ingates.inputs.twin.right;

	GATE* gleft = &(m_vGates[idleft]);
	GATE* gright = &(m_vGates[idright]);
	uint32_t ttable = gate->gs.ttable;

	InstantiateGate(gate);

	for (uint32_t g = 0; g < gate->nvals; g++) {
		GarbleUniversalGate(gate, g, gleft, gright, ttable);
		m_nUniversalGateTableCtr++;
		//gate->gs.yinput.pi[g] = 0;
		assert(gate->gs.yinput.pi[g] < 2);
	}
	UsedGate(idleft);
	UsedGate(idright);

	return true;
}

void PRFServerSharing::GarbleUniversalGate(GATE* ggate, uint32_t pos, GATE* gleft, GATE* gright, uint32_t ttable) {
	BYTE* univ_table = m_vUniversalGateTable.GetArr() + m_nUniversalGateTableCtr * KEYS_PER_UNIV_GATE_IN_TABLE * m_nSecParamBytes;
	uint32_t ttid = (gleft->gs.yinput.pi[pos] << 1) + gright->gs.yinput.pi[pos];

	assert(gright->instantiated && gleft->instantiated);

	uint8_t* bLMaskBuf[2] = { gleft->gs.yinput.outKey[0] + pos * m_nSecParamBytes, gleft->gs.yinput.outKey[1] + pos * m_nSecParamBytes };
	uint8_t* bRMaskBuf[2] = { gright->gs.yinput.outKey[0] + pos * m_nSecParamBytes, gright->gs.yinput.outKey[1] + pos * m_nSecParamBytes };

	BYTE* outkey[2];
	outkey[0] = ggate->gs.yinput.outKey[0] + pos * m_nSecParamBytes;
	outkey[1] = ggate->gs.yinput.outKey[1] + pos * m_nSecParamBytes;

	assert(((uint64_t*)m_bZeroBuf)[0] == 0);
	//GRR: Encryption with both original keys of a zero-string becomes the key on the output wire of the gate
	EncryptWireGRR3(outkey[0], m_bZeroBuf, bLMaskBuf[0], bRMaskBuf[0], 0);

	//Sort the values according to the permutation bit and precompute the second wire key
	BYTE kbit = outkey[0][m_nSecParamBytes - 1] & 0x01;
	ggate->gs.yinput.pi[pos] = ((ttable >> ttid) & 0x01) ^ kbit;//((kbit^1) & (ttid == 3)) | (kbit & (ttid != 3));

#ifdef DEBUGYAOSERVER
	std::cout << " encrypting : ";
	PrintKey(m_bZeroBuf);
	std::cout << " using: ";
	PrintKey(m_bLMaskBuf[0]);
	std::cout << " (" << (uint32_t)gleft->gs.yinput.pi[pos] << ") and : ";
	PrintKey(m_bRMaskBuf[0]);
	std::cout << " (" << (uint32_t)gright->gs.yinput.pi[pos] << ") to : ";
	PrintKey(m_bOKeyBuf[0]);
	std::cout << std::endl;
#endif
	memcpy(outkey[kbit], outkey[0], m_nSecParamBytes);
	m_cCrypto->gen_rnd(outkey[kbit ^ 1], m_nSecParamBytes);

	for (uint32_t i = 1, keyid; i < 4; i++, univ_table += m_nSecParamBytes) {
		keyid = ((ttable >> (ttid ^ i)) & 0x01) ^ ggate->gs.yinput.pi[pos];
		assert(keyid < 2);
		//cout << "Encrypting into outkey = " << outkey << ", " << (unsigned long) m_bOKeyBuf[0] << ", " <<  (unsigned long) m_bOKeyBuf[1] <<
		//		", truthtable = " << (unsigned uint32_t) g_TruthTable[id^i] << ", mypermbit = " << (unsigned uint32_t) ggate->gs.yinput.pi[pos] << ", id = " << id << endl;
		EncryptWireGRR3(univ_table, outkey[keyid], bLMaskBuf[i >> 1], bRMaskBuf[i & 0x01], i);
#ifdef DEBUGYAOSERVER
		std::cout << " encrypting : ";
		PrintKey(m_bOKeyBuf[0]); // TODO: check that we print the right value
		std::cout << " using: ";
		PrintKey(m_bLMaskBuf[i >> 1]);
		std::cout << " (" << (uint32_t)gleft->gs.yinput.pi[pos] << ") and : ";
		PrintKey(m_bRMaskBuf[i & 0x01]);
		std::cout << " (" << (uint32_t)gright->gs.yinput.pi[pos] << ") to : ";
		PrintKey(univ_table); // TODO: check that we print the right value
		std::cout << std::endl;
#endif
	}
}