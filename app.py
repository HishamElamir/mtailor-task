import torch
from PIL import Image
from torch import argmax
import numpy as np
import torchvision.transforms as transforms
import onnxruntime as onnxrt

from test_onnx import to_numpy, preprocess_numpy

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    onnx_session= onnxrt.InferenceSession("mtailor-resnet.onnx")

    img1 = Image.open("n01440764_tench.jpeg")
    img1 = preprocess_numpy(img1)
    img2 = Image.open("n01667114_mud_turtle.JPEG")
    img2 = preprocess_numpy(img2)

    onnx_inputs= {
        onnx_session.get_inputs()[0].name: to_numpy(img1)
    }
    onnx_output = onnx_session.run(None, onnx_inputs)
    img_label = onnx_output[0]
    assert np.argmax(img_label) == 0

    onnx_inputs= {
        onnx_session.get_inputs()[0].name: to_numpy(img2)
    }
    onnx_output = onnx_session.run(None, onnx_inputs)
    img_label = onnx_output[0]
    assert np.argmax(img_label) == 35

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    img1 = Image.open(prompt)
    img1 = preprocess_numpy(img1)
    # Run the model
    onnx_inputs= {
        onnx_session.get_inputs()[0].name: to_numpy(img1)
    }
    onnx_output = onnx_session.run(None, onnx_inputs)
    img_label = onnx_output[0]    

    # Return the results as a dictionary
    return np.argmax(img_label)
