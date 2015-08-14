#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// #include <iomanip>    // ywc //
// using namespace std;  // ywc //

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // add more here
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
    << "The data and label should have the same number of instances";
  /*CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))
    << "The data and label should have the same height";
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3))
    << "The data and label should have the same width";*/
  // Currently only supports scalar inputs
  CHECK_EQ(bottom[0]->shape(1),1) << "Currently only supports scalar inputs";
  CHECK_EQ(bottom[0]->shape(2),1) << "Currently only supports scalar inputs";
  CHECK_EQ(bottom[0]->shape(3),1) << "Currently only supports scalar inputs";
  CHECK_EQ(bottom[1]->num_axes(),1) << "Currently only supports scalar inputs";
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // assume scalar inputs
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int N = bottom[0]->shape(0);
  for (int i = 0; i < N; ++i) {
    const int label = static_cast<int>(bottom_label[i]);
    // Binary label check --- 1: positive, 0: negative
    //   Has to be consistent with loss computation in sigmoid_cross_entropy_loss_layer.cpp
    DCHECK_GE(label, 0);
    DCHECK_LT(label, 1);
    if (label == 1) {
      accuracy += bottom_data[i] >= 0;
    }
    if (label == 0) {
      accuracy += bottom_data[i] < 0;
    }
    // print label & scores
    /*LOG(INFO) << "  "
      << "ind: " << setfill('0') << setw(2) << i << "  "
      << "label: " << setfill('0') << setw(3) << label << "  "
      << "score0: " << setfill('0') << setw(3) << bottom_data[i] << "  "
      << "accuracy: " << accuracy;*/
  }
  top[0]->mutable_cpu_data()[0] = accuracy / N;
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe
