#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void BasePrefetchingDataPyramidLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  for (int i = 0; i < top.size()-1; ++i) {
    // Reshape to loaded data.
    top[i]->ReshapeLike(*this->prefetch_data_vector_[i]);
    // Copy the data
    caffe_copy(prefetch_data_vector_[i]->count(), prefetch_data_vector_[i]->cpu_data(),
        top[i]->mutable_gpu_data());
  }
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[top.size()-1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[top.size()-1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataPyramidLayer);

}  // namespace caffe
