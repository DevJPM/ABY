#ifndef __AES_PROCESSOR_H__
#define __AES_PROCESSOR_H__

#include <cstdint>
#include <cstddef>

using std::size_t;

class AESProcessorHalfGateGarbling
{
public:
	/**
	* Fills an otherwise specified buffer with AES PRF calculations for AND gates
	* \param baseOffset the offset into the queue of the AND gates
	* \param tableCounter the number of tables already generated
	* \param numTablesInBatch the number of garbled tables for which to calculate the PRF evaluations
	*/
	virtual void fillAESBufferAND(size_t baseOffset, uint32_t tableCounter, size_t numTablesInBatch) = 0;

	/**
	* Sets the global key difference R
	* \param r the global key difference
	*/
	virtual void setGlobalKey(const uint8_t* r) = 0;

	virtual ~AESProcessorHalfGateGarbling() {};
};

class AESProcessorHalfGateEvaluation
{
public:
	/**
	* Performs the two AES evaluations per AND gate necessary and writes the result into the right field of the gate
	* \param tableCounter the number of garbled tables inspected at this point
	*/
	virtual void computeAESPreOutKeys(uint32_t tableCounter, size_t numTablesInBatch) = 0;

	virtual ~AESProcessorHalfGateEvaluation() {};
};

#endif