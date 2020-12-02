#ifndef __AES_PROCESSOR_H__
#define __AES_PROCESSOR_H__

#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>

#include "../../circuit/abycircuit.h"

using std::size_t;

class VectorizedQueueProcessor
{
protected:
	void ProcessQueue(const std::vector<GATE*>& queue, const size_t vectorWidth, const size_t numWires, uint32_t wireCounter, uint8_t* tableBuffer);
	virtual void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, uint8_t* tableBuffer) = 0;
	virtual void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) = 0;
};

class AESProcessor : protected VectorizedQueueProcessor
{
public:
	/**
	* Performs the two AES evaluations per AND gate necessary and writes the result into the right field of the gate
	* \param tableCounter the number of garbled tables inspected at this point
	*/
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) = 0;

	virtual ~AESProcessor() {};
};

class AESProcessorHalfGateGarbling : public AESProcessor
{
public:
	/**
	* Sets the global key difference R
	* \param r the global key difference
	*/
	virtual void setGlobalKey(const uint8_t* r) = 0;

	virtual ~AESProcessorHalfGateGarbling() {};
};

#endif