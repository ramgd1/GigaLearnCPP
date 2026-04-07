#include "ExperienceBuffer.h"
#include <numeric>

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {

	// Indices on same device as data for index_select (required when data is on CUDA)
	Tensor tIndices = torch::tensor(IList(indices, indices + size), torch::TensorOptions().dtype(torch::kLong).device(data.states.device()));

	ExperienceTensors result;
	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		*toItr = torch::index_select(*fromItr, 0, tIndices);

	return result;
}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(torch::Tensor indices) const {

	ExperienceTensors result;
	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		*toItr = torch::index_select(*fromItr, 0, indices);

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	int64_t expSize = static_cast<int64_t>(data.states.size(0));

	int64_t* indices = new int64_t[expSize];
	std::iota(indices, indices + expSize, 0);
	std::shuffle(indices, indices + expSize, rng);

	auto dev = data.states.device();
	// One index tensor on device per epoch; batches use slices (avoids 6 index tensors per batch)
	Tensor tIndices = torch::from_blob(indices, { expSize }, torch::TensorOptions().dtype(torch::kLong)).clone().to(dev);
	delete[] indices;

	std::vector<ExperienceTensors> result;
	for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {

		int64_t curBatchSize = batchSize;
		if (startIdx + batchSize * 2 > expSize) {
			if (overbatching) {
				curBatchSize = expSize - startIdx;
			}
		}

		result.push_back(_GetSamples(tIndices.slice(0, startIdx, startIdx + curBatchSize)));
	}

	return result;
}