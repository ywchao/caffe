#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
DataPyramidLayer<Dtype>::~DataPyramidLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void DataPyramidLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }
  // Read a data point, to initialize the prefetch and top blobs.
  Datum datum;
  datum.ParseFromString(cursor_->value());
  // Use data_transformer to infer the expected blob shape from datum.
  CHECK_GE(top.size()-1, 1);
  CHECK_LE(top.size()-1, MaxTopBlobs());
  vector<vector<int> > top_shape_vector(top.size()-1,vector<int>(4));
  this->data_transformer_->InferBlobShape(datum, top_shape_vector);

  // Create prefetch data and tranformed data blobs and reshape top[0] to
  // top[top.size()-1].
  this->prefetch_data_vector_.resize(top.size()-1);
  this->transformed_data_vector_.resize(top.size()-1);
  for (int i = 0; i < top.size()-1; ++i) {
    this->transformed_data_vector_[i] = new Blob<Dtype>();
    this->transformed_data_vector_[i]->Reshape(top_shape_vector[i]);
    top_shape_vector[i][0] = this->layer_param_.data_param().batch_size();
    this->prefetch_data_vector_[i] = new Blob<Dtype>();
    this->prefetch_data_vector_[i]->Reshape(top_shape_vector[i]);
    top[i]->ReshapeLike(*this->prefetch_data_vector_[i]);

    LOG(INFO) << "output data " << i+1 << " size: " << top[0]->num() << ","
        << top[i]->channels() << "," << top[i]->height() << ","
        << top[i]->width();
  }
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, this->layer_param_.data_param().batch_size());
    top[top.size()-1]->Reshape(label_shape);
    this->prefetch_label_.Reshape(label_shape);
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataPyramidLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  const int top_data_size = this->prefetch_data_vector_.size();
  CHECK_EQ(top_data_size, this->transformed_data_vector_.size());
  for (int i = 0; i < top_data_size; ++i) {
    CHECK(this->prefetch_data_vector_[i]->count());
    CHECK(this->transformed_data_vector_[i]->count());
  }

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum datum;
  datum.ParseFromString(cursor_->value());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<vector<int> > top_shape_vector(top_data_size,vector<int>(4));
  this->data_transformer_->InferBlobShape(datum, top_shape_vector);
  // Reshape transformed data and prefetch data
  for (int i = 0; i < top_data_size; ++i) {
    this->transformed_data_vector_[i]->Reshape(top_shape_vector[i]);
    top_shape_vector[i][0] = this->layer_param_.data_param().batch_size();
    this->prefetch_data_vector_[i]->Reshape(top_shape_vector[i]);
  }

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  timer.Start();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a datum
    Datum datum;
    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    for (int i = 0; i < top_data_size; ++i) {
      Dtype* top_data_i = this->prefetch_data_vector_[i]->mutable_cpu_data();
      int offset = this->prefetch_data_vector_[i]->offset(item_id);
      this->transformed_data_vector_[i]->set_cpu_data(top_data_i + offset);
    }
    this->data_transformer_->Transform(datum, this->transformed_data_vector_);
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();
    timer.Start();
    // go to the next item.
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataPyramidLayer);
REGISTER_LAYER_CLASS(DataPyramid);

}  // namespace caffe
