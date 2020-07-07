import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
# import tensorrt as trt
# from yolov3trt import common

# TRT_LOGGER = trt.Logger()

class TestUpsample(nn.Module):
    def __int__(self):
        super(TestUpsample, self).__init__()

    def forward(self, x):
        x = nn.Upsample(scale_factor=2, mode="nearest")(x)
        return x

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode = 'nearest')
        return x

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1,3,320,320]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main(dummy_input):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = "upsample-1.5.0.onnx"
    engine_file_path = "upsample-1.5.0.trt"
    # Download a dog image and save it to the following file path:

    image = dummy_input.numpy()

    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        inputs[0].host = image
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    return trt_outputs

if __name__ == "__main__":
    image = cv2.imread("test.jpg")[:,:,::-1] # 1,3,320,320
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as
    image = np.array(image, dtype=np.float32, order='C')
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    torch_model = TestUpsample()
    output = torch_model(image)

    # onnx 1.5.0 版本的转换
    # torch_out = torch.onnx._export(torch_model, image, 'upsample-1.5.0.onnx', verbose=True) # 1,3,640,640
    # onnx 1.6.0 版本的转换
    torch_out = torch.onnx._export(torch_model, image, 'upsample-1.6.0.onnx', verbose=True, opset_version=11)
    # np.testing.assert_almost_equal(output.data.cpu().numpy(), torch_out.data.cpu().numpy(), decimal=3)
    #
    # torch_out = torch_out.squeeze().data.cpu().numpy()
    # torch_out = np.transpose(torch_out, [1, 2, 0])[:,:,::-1]
    # cv2.imwrite("torch_out.jpg", torch_out)

    # trt_outputs = main(image)[0].reshape(1,3,640,640)
    # np.testing.assert_almost_equal(output.data.cpu().numpy(), trt_outputs, decimal=3)
    #
    # trt_outputs = trt_outputs.squeeze()
    # trt_outputs = np.transpose(trt_outputs, [1, 2, 0])[:,:,::-1]
    # cv2.imwrite("trt_out.jpg", trt_outputs)




