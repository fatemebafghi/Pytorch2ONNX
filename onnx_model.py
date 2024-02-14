import onnxruntime as ort
from preprocess import  classifier_preprocess
from PIL import Image
import numpy as np
from logzero import logger
import time


class onnx_model():

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.create_session()

    def create_session(self):
        session_option = ort.SessionOptions()
        session_option.log_severity_level = 2
        session_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider = [('CUDAExecutionProvider', {"device_id": 0,
                                               'arena_extend_strategy': 'kSameAsRequested'}),
                    'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_path, sess_options=session_option, providers=provider)

        logger.info(f"execution provider's option = {self.session.get_provider_options()}")

    def __call__(self, input_image):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        features = self.session.run(output_names=[output_name], input_feed={input_name: input_image})
        label = np.argmax(features)
        logger.info(f"label = {label}")


if __name__ == '__main__':

    onnx_vit = onnx_model("model.onnx")
    image_path = "provide your test image path"
    image = Image.open(image_path).convert("RGB")
    image = classifier_preprocess(image)
    start_time = time.time()
    result = onnx_vit(image)

    logger.info(f"executed in {time.time() - start_time }secends")