#include "halfgates_prp_server.h"
#include "../../aby/abysetup.h"
#include "../aes_processors/aesni_halfgate_processors.h"
#include "../aes_processors/vaes_halfgate_processors.h"
#include "../cpu_features/include/cpuinfo_x86.h"

static const cpu_features::X86Features CPU_FEATURES = cpu_features::GetX86Info().features;

void HalfGatesPRPServerSharing::prepareGarblingSpecificSetup() {
	m_vR.Create(m_cCrypto->get_seclvl().symbits, m_cCrypto);
	m_vR.SetBit(m_cCrypto->get_seclvl().symbits - 1, 1);

	m_aesProcessor->setGlobalKey(m_vR.GetArr()); 
}

bool HalfGatesPRPServerSharing::evaluateConstantGate(GATE* gate) {
	//assign 0 and 1 gates
	UGATE_T constval = gate->gs.constval;
	InstantiateGate(gate);
	memset(gate->gs.yinput.outKey[0], 0, m_nSecParamBytes * gate->nvals);
	for (uint32_t i = 0; i < gate->nvals; ++i) {
		memcpy(gate->gs.yinput.outKey[1] + m_nSecParamBytes * i, m_vR.GetArr(), m_nSecParamBytes);
		if (constval == 1L) {
			gate->gs.yinput.pi[i] = 1;
		}
		else {
			gate->gs.yinput.pi[i] = 0;
		}
	}

	return true;
}

void HalfGatesPRPServerSharing::createOppositeInputKeys(CBitVector& oppositeInputKeys, CBitVector& regularInputKeys, size_t numKeys)
{
	const size_t numBytes = numKeys * m_nSecParamBytes;
	
	BYTE* buffer = (BYTE*)malloc(numBytes);
	oppositeInputKeys.AttachBuf(buffer, numBytes);

	for (size_t i = 0; i < numBytes / m_nSecParamBytes; ++i)
	{
		m_pKeyOps->XOR(oppositeInputKeys.GetArr() + i * m_nSecParamBytes, regularInputKeys.GetArr() + i * m_nSecParamBytes, m_vR.GetArr());
	}
}

void HalfGatesPRPServerSharing::evaluateDeferredANDGates(ABYSetup* setup, size_t numWires)
{
	// the buffers are needed for the batch processing
	for (auto* currentGate : getAndQueue())
		InstantiateGate(currentGate);

	// we call into another class here as this allows us to exploit dynamic dispatch
	// to switch between AES256, AES128, AES-NI and VAES as needed based on a ctor parameter
	m_aesProcessor->computeOutKeysAndTable(m_nAndGateTableCtr, numWires, m_vAndGateTable.GetArr());

	for (auto* currentGate : getAndQueue())
	{
		uint32_t idleft = currentGate->ingates.inputs.twin.left; //gate->gs.ginput.left;
		uint32_t idright = currentGate->ingates.inputs.twin.right; //gate->gs.ginput.right;

		m_nAndGateTableCtr += currentGate->nvals;

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

bool HalfGatesPRPServerSharing::evaluateXORGate(GATE* gate)
{
	uint32_t idleft = gate->ingates.inputs.twin.left; //gate->gs.ginput.left;
	uint32_t idright = gate->ingates.inputs.twin.right; //gate->gs.ginput.right;

	BYTE* lpi = m_vGates[idleft].gs.yinput.pi;
	BYTE* rpi = m_vGates[idright].gs.yinput.pi;

	BYTE* lkey = m_vGates[idleft].gs.yinput.outKey[0];
	BYTE* rkey = m_vGates[idright].gs.yinput.outKey[0];
	InstantiateGate(gate);

	BYTE* gpi = gate->gs.yinput.pi;
	BYTE* gkey[] = { gate->gs.yinput.outKey[0], gate->gs.yinput.outKey[1] };

#ifdef GATE_INST_FLAG
	assert(m_vGates[idleft].instantiated);
	assert(m_vGates[idright].instantiated);
#endif
	for (uint32_t g = 0; g < gate->nvals; g++, gpi++, lpi++, rpi++, lkey += m_nSecParamBytes, rkey += m_nSecParamBytes, gkey[0] += m_nSecParamBytes, gkey[1] += m_nSecParamBytes) {
		*gpi = *lpi ^ *rpi;
		m_pKeyOps->XOR(gkey[0], lkey, rkey);
		m_pKeyOps->XOR(gkey[1], gkey[0], m_vR.GetArr());
		assert(*gpi < 2);
	}

#ifdef DEBUGYAOSERVER
	PrintKey(gate->gs.yinput.outKey);
	std::cout << " (" << (uint32_t)gate->gs.yinput.pi[0] << ") = ";
	PrintKey(m_vGates[idleft].gs.yinput.outKey);
	std::cout << " (" << (uint32_t)m_vGates[idleft].gs.yinput.pi[0] << ")(" << idleft << ") ^ ";
	PrintKey(m_vGates[idright].gs.yinput.outKey);
	std::cout << " (" << (uint32_t)m_vGates[idright].gs.yinput.pi[0] << ")(" << idright << ")" << std::endl;
#endif

	assert(m_vGates[idleft].gs.yinput.pi[0] < 2 && m_vGates[idright].gs.yinput.pi[0] < 2);
	UsedGate(idleft);
	UsedGate(idright);

	return true;
}

bool HalfGatesPRPServerSharing::evaluateUNIVGate(GATE* gate)
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

void HalfGatesPRPServerSharing::GarbleUniversalGate(GATE* ggate, uint32_t pos, GATE* gleft, GATE* gright, uint32_t ttable) {
	BYTE* univ_table = m_vUniversalGateTable.GetArr() + m_nUniversalGateTableCtr * KEYS_PER_UNIV_GATE_IN_TABLE * m_nSecParamBytes;
	uint32_t ttid = (gleft->gs.yinput.pi[pos] << 1) + gright->gs.yinput.pi[pos];

	assert(gright->instantiated && gleft->instantiated);

	memcpy(m_bLMaskBuf[0].get(), gleft->gs.yinput.outKey[0] + pos * m_nSecParamBytes, m_nSecParamBytes);
	m_pKeyOps->XOR(m_bLMaskBuf[1].get(), m_bLMaskBuf[0].get(), m_vR.GetArr());

	memcpy(m_bRMaskBuf[0].get(), gright->gs.yinput.outKey[0] + pos * m_nSecParamBytes, m_nSecParamBytes);
	m_pKeyOps->XOR(m_bRMaskBuf[1].get(), m_bRMaskBuf[0].get(), m_vR.GetArr());

	BYTE* outkey[2];
	outkey[0] = ggate->gs.yinput.outKey[0] + pos * m_nSecParamBytes;
	outkey[1] = ggate->gs.yinput.outKey[1] + pos * m_nSecParamBytes;

	assert(((uint64_t*)m_bZeroBuf)[0] == 0);
	//GRR: Encryption with both original keys of a zero-string becomes the key on the output wire of the gate
	EncryptWireGRR3(outkey[0], m_bZeroBuf, m_bLMaskBuf[0].get(), m_bRMaskBuf[0].get(), 0);

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
	m_pKeyOps->XOR(outkey[kbit ^ 1], outkey[kbit], m_vR.GetArr());

	for (uint32_t i = 1, keyid; i < 4; i++, univ_table += m_nSecParamBytes) {
		keyid = ((ttable >> (ttid ^ i)) & 0x01) ^ ggate->gs.yinput.pi[pos];
		assert(keyid < 2);
		//cout << "Encrypting into outkey = " << outkey << ", " << (unsigned long) m_bOKeyBuf[0] << ", " <<  (unsigned long) m_bOKeyBuf[1] <<
		//		", truthtable = " << (unsigned uint32_t) g_TruthTable[id^i] << ", mypermbit = " << (unsigned uint32_t) ggate->gs.yinput.pi[pos] << ", id = " << id << endl;
		EncryptWireGRR3(univ_table, outkey[keyid], m_bLMaskBuf[i >> 1].get(), m_bRMaskBuf[i & 0x01].get(), i);
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
	m_pKeyOps->XOR(outkey[1], outkey[0], m_vR.GetArr());
}

void HalfGatesPRPServerSharing::InitServer()
{
	m_bLMaskBuf.emplace_back(static_cast<uint8_t*>(std::aligned_alloc(16, m_nSecParamBytes)));
	m_bLMaskBuf.emplace_back(static_cast<uint8_t*>(std::aligned_alloc(16, m_nSecParamBytes)));
	m_bRMaskBuf.emplace_back(static_cast<uint8_t*>(std::aligned_alloc(16, m_nSecParamBytes)));
	m_bRMaskBuf.emplace_back(static_cast<uint8_t*>(std::aligned_alloc(16, m_nSecParamBytes)));

	if (m_nSecParamBytes != 16)
	{
		std::cerr << "unsupported security parameter." << std::endl;
		assert(false);
	}
	else
	{
		if (CPU_FEATURES.vaes && CPU_FEATURES.avx512f)
			m_aesProcessor = std::make_unique<FixedKeyLTGarblingVaesProcessor>(getAndQueue(), m_vGates);
		else if (CPU_FEATURES.aes && CPU_FEATURES.sse4_1)
			m_aesProcessor = std::make_unique<FixedKeyLTGarblingAesniProcessor>(getAndQueue(), m_vGates);
		else
		{
			std::cerr << "unsupported host CPU." << std::endl;
			assert(false);
		}
	}
}
