#ifndef __AESNI_HALFGATE_PROCESSORS_H__
#define __AESNI_HALFGATE_PROCESSORS_H__

#include <vector>
#include <memory>
#include <cstdint>

using std::uint8_t;

#include "../../circuit/abycircuit.h"
#include "aes_processor.h"
#include "../../ABY_utils/memory.h"
#include "../yaoserversharing.h"

class FixedKeyProvider final
{
public:
	FixedKeyProvider() :
		m_expandedStaticAESKey(static_cast<uint8_t*>(std::aligned_alloc(16, 11 * 16)))
	{
		expandAESKey();
	}
	const uint8_t* getExpandedStaticKey() const { return m_expandedStaticAESKey.get(); }
private:
	void expandAESKey();
	std::unique_ptr<uint8_t[], free_byte_deleter> m_expandedStaticAESKey;
};

class InputKeyLTGarblingAesniProcessor : public AESProcessorHalfGateGarbling
{
public:
	InputKeyLTGarblingAesniProcessor(uint8_t* buffer, size_t bufferSize, const std::vector<YaoServerSharing::GarbledTableJob>& tableGateQueue, const std::vector<GATE>& vGates) :
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

	const std::vector<YaoServerSharing::GarbledTableJob>& m_tableGateQueue;
	const std::vector<GATE>& m_vGates;
	const uint8_t* m_globalRandomOffset;
	uint8_t* m_aesBuffer;
	const size_t m_bufferSize;
};

class FixedKeyLTGarblingAesniProcessor : public AESProcessorHalfGateGarbling
{
public:
	FixedKeyLTGarblingAesniProcessor(uint8_t* buffer, size_t bufferSize, const std::vector<YaoServerSharing::GarbledTableJob>& tableGateQueue,const std::vector<GATE>& vGates) :
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
	template<size_t width> void fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch,size_t bufferOffset);

	FixedKeyProvider m_fixedKeyProvider;
	const std::vector<YaoServerSharing::GarbledTableJob>& m_tableGateQueue;
	const std::vector<GATE>& m_vGates;
	const uint8_t* m_globalRandomOffset;
	uint8_t* m_aesBuffer;
	const size_t m_bufferSize;
};

class FixedKeyLTEvaluatingAesniProcessor : public AESProcessorHalfGateEvaluation
{
public:
	FixedKeyLTEvaluatingAesniProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates) :
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

class InputKeyLTEvaluatingAesniProcessor : public AESProcessorHalfGateEvaluation
{
public:
	InputKeyLTEvaluatingAesniProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates) :
		m_gateQueue(gateQueue),
		m_vGates(vGates)
	{
	}
	virtual void computeAESPreOutKeys(uint32_t tableCounter, size_t numTablesInBatch) override;
private:
	template<size_t width>  void computeAESPreOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch);

	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;
};


#endif