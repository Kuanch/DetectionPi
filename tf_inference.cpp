#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

using namespace tensorflow;


Status ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto root = tensorflow::Scope::NewRootScope();

  string input_name = "file_reader";
  string output_name = "normalized";
  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name),
                                               file_name);
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;

  image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));


  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  return Status::OK();
}




int main(int argc, char* argv[]) 
{
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  std::cout << "Session build successfully!" << "\n";

  GraphDef graph_def;
  std::string model_path(argv[2]);
  status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  std::cout << "Graph build successfully!" << "\n";

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  std::cout << "Add graph to session successfully!" << "\n";  


  std::vector<tensorflow::Tensor> inputs;
  std::string image_path(argv[1]);

  int32 input_dim = 300;
  int32 input_mean = 128;
  int32 input_std = 128;
  if (!ReadTensorFromImageFile(image_path, input_dim, input_dim, input_mean,
                               input_std, &inputs).ok()) {
    LOG(ERROR) << "Load image";
    return -1;
  }
  std::cout << "Read images successfully!" << "\n";  


  std::vector<tensorflow::Tensor> outputs;
  string input_layer = "image_tensor:0";
  string output_layer = {"detection_boxes:0"};
  if (!session->Run({{input_layer, inputs[0]}},
                     {output_layer}, {}, &outputs).ok()) {
    LOG(ERROR) << "Running model failed";
    return -1;
  }

  ///home/sixigma/face_detection_model/data/img2.jpg
  //array([0.23533039, 0.08384116, 0.38354808, 0.16164924], dtype=float32)
}