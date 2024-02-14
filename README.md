How to convert a pytorch model like timm to onnx

# Image Classification ONNX Model
This repository contains Python code to convert a PyTorch image classification model to ONNX format and run inferences using the ONNX model.

## Files
1. convert2onnx.py: Script to trace the PyTorch model and export it to ONNX format. Saves the ONNX model to model.onnx.
2. onnx_model.py:  Loads the ONNX model and sets up an InferenceSession. Takes an input image, preprocesses it, runs inference, and prints the predicted label.
3. preprocess.py: Contains preprocessing code to format images before passing to the model. Resizes to 224x224, normalizes using ImageNet mean and std dev, converts to NCHW format.

## Usage
Convert PyTorch model to ONNX:
````
python convert2onnx.py
````

Run inference using ONNX model:
````
python onnx_model.py
````
Update image_path in onnx_model.py to the path of the image you want to classify.

## Requirements
1. PyTorch 
2. ONNX 
3. ONNX Runtime 
4. OpenCV 
5. TIMM (for model creation)

## Model
The example model used is a ViT-Large model pretrained on ImageNet-21k from the TIMM library. It is converted to an 11-class classifier.

for more information please visit https://medium.com/@fatemebfg/how-to-convert-a-pytorch-model-to-onnx-8ff347cf5113