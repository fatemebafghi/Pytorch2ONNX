import torch
import timm
from torch.jit import trace
from logzero import logger
from torch import onnx as torch_onnx
import onnx

class Compile():

    def __init__(self):
        self.onnx_path = "model.onnx"
        self.device = "cuda"
        self.check_point_path = "/home/sd/Work/Projects/ONNX_Tutorial/best_model.pth"
        self.load_model()
        self.convert()

    def load_model(self):

        self.original_model = timm.create_model("vit_large_patch16_224_in21k",
                                                pretrained=False,
                                                checkpoint_path=self.check_point_path,
                                                num_classes=11).to(self.device)

        with torch.no_grad():
            self.original_model.eval()
            self.example_input = torch.rand(1, 3, 224, 224).to(self.device)
            self.scripted_model = trace(self.original_model, example_inputs=self.example_input, check_trace=False)

            logger.info("model was traced successfuly")

    def convert(self):
        input_names = ['input']
        output_name = ['output']

        with torch.no_grad():

            self.scripted_model.eval()

            dynamic_axix = {"input": {0: "batch", 2: "width", 3: "height"},
                            "output": {0: "batch"}}
            torch_onnx.export(self.scripted_model,
                              self.example_input,
                              self.onnx_path,
                              verbose=True,
                              do_constant_folding=True,
                              dynamic_axes=dynamic_axix,
                              input_names=input_names,
                              output_names=output_name
                              )

        self.check_onnx()

    def check_onnx(self):

        model = onnx.load_model(self.onnx_path)

        try:
            onnx.checker.check_model(model)
        except Exception as e:
            logger.error(f"{e}")
        else:
            logger.info("passed successsfully")









if __name__ == "__main__":

    compiler = Compile()
