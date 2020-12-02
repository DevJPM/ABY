#include "aes_processor.h"

#include <cassert>

void VectorizedQueueProcessor::ProcessQueue(const std::vector<GATE*>& queue, const size_t vectorWidth, const size_t numWires, uint32_t wireCounter, uint8_t* tableBuffer)
{
	assert(numWires >= queue.size());
	if (queue.size() == 0)
		return;

	const size_t leftovers = numWires % vectorWidth;
	const size_t mainBulkSize = numWires - leftovers;

	BulkProcessor(wireCounter,mainBulkSize,tableBuffer);

	size_t numWiresLeft = 0;
	int64_t ridx;

	for (ridx = queue.size() - 1; ridx >= 0; --ridx)
	{
		numWiresLeft += queue[ridx]->nvals;
		if (numWiresLeft >= leftovers)
			break;
	}

	if (leftovers > 0)
	{
		LeftoversProcessor(wireCounter + mainBulkSize, leftovers, ridx, numWiresLeft - leftovers, tableBuffer);
	}
}
