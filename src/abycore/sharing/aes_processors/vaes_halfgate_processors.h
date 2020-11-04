#ifndef __VAES_HALFGATE_PROCESSORS_H__
#define __VAES_HALFGATE_PROCESSORS_H__

#include <vector>
#include <memory>
#include <cstdint>

using std::uint8_t;

#include "../../circuit/abycircuit.h"
#include "aes_processor.h"
#include "../../ABY_utils/memory.h"
#include "../yaoserversharing.h"
#include "aesni_halfgate_processors.h"

class FixedKeyLTGarblingVaesProcessor : public AESProcessorHalfGateGarbling
{
public:
	FixedKeyLTGarblingVaesProcessor(uint8_t* buffer, size_t bufferSize, const std::vector<YaoServerSharing::GarbledTableJob>& tableGateQueue, const std::vector<GATE>& vGates) :
		m_tableGateQueue(tableGateQueue),
		m_vGates(vGates),
		m_aesBuffer(buffer),
		m_bufferSize(bufferSize)
	{
	}
	virtual void setGlobalKey(const uint8_t* r) override { m_globalRandomOffset = r; }
	virtual void fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch) override;
private:
	// only processes multiples of width
	template<size_t width> void fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch, size_t bufferOffset);

	FixedKeyProvider m_fixedKeyProvider;
	const std::vector<YaoServerSharing::GarbledTableJob>& m_tableGateQueue;
	const std::vector<GATE>& m_vGates;
	const uint8_t* m_globalRandomOffset;
	uint8_t* m_aesBuffer;
	const size_t m_bufferSize;
};

class FixedKeyLTEvaluatingVaesProcessor : public AESProcessorHalfGateEvaluation
{
public:
	FixedKeyLTEvaluatingVaesProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates) :
		m_gateQueue(gateQueue),
		m_vGates(vGates)
	{
	}
	virtual void computeAESPreOutKeys(uint32_t tableCounter, size_t numTablesInBatch) override;
private:
	template<size_t width>  void computeAESPreOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch);

	FixedKeyProvider m_fixedKeyProvider;
	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;
};

#endif