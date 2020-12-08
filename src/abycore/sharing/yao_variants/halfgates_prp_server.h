#ifndef __HALFGATES_PRP_SERVER_H__
#define __HALFGATES_PRP_SERVER_H__

#include "../yaoserversharing.h"
#include <vector>

class HalfGatesPRPServerSharing : public YaoServerSharing {
public:
	/** Constructor of the class.*/
	HalfGatesPRPServerSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
		YaoServerSharing(context, role, sharebitlen, circuit, crypt, circdir)
	{
		InitServer();
	}
	/** Destructor of the class.*/
	//~HalfGatesPRPClientSharing();
protected:
	size_t ciphertextPerAND() const override { return 2; }
	size_t ciphertextPerXOR() const override { return 0; }
	void prepareGarblingSpecificSetup() override;

	void evaluateDeferredXORGates(size_t numWires) override {}
	void evaluateDeferredANDGates(ABYSetup* setup, size_t numWires) override;
	bool evaluateXORGate(GATE* gate) override;
	bool evaluateANDGate(ABYSetup* setup, GATE* gate) override { return false; }
	bool evaluateUNIVGate(GATE* gate) override;
	bool evaluateConstantGate(GATE* gate) override;
	bool evaluateInversionGate(GATE* gate) override;
	void resetGarblingSpecific() override { m_vR.delCBitVector(); }
	void createOppositeInputKeys(CBitVector& oppositeInputKeys, CBitVector& reglarInputKeys, size_t numKeys) override;
	void copyServerInputKey(uint8_t inputBit, uint8_t permutationBit, size_t targetByteOffset, size_t sourceByteOffset) override;
	uint8_t computePermutationValueFromBoolConv(uint8_t inputBit, uint8_t permutationBit) override { return inputBit ^ permutationBit; }
private:
	void InitServer();
	void GarbleUniversalGate(GATE* ggate, uint32_t pos, GATE* gleft, GATE* gright, uint32_t ttable);

	std::unique_ptr<AESProcessorHalfGateGarbling> m_aesProcessor;
	std::vector<std::unique_ptr<uint8_t[], free_byte_deleter>> m_bRMaskBuf;
	std::vector<std::unique_ptr<uint8_t[], free_byte_deleter>> m_bLMaskBuf;
	
	//Global constant key
	CBitVector m_vR; /**< _____________*/
};

#endif