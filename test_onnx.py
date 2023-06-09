# IMPORT
from PIL import Image
from torch import argmax
import numpy as np
import torchvision.transforms as transforms
import onnxruntime as onnxrt

# TRANSFORM INPUT/OUTPUT TO NUMPY
def to_numpy(tensor):
    return tensor.detach().cpu().numpy()\
        if tensor.requires_grad else tensor.cpu().numpy()

# TRANSFORMING IMAGE TO BE VALID FOR MODEL
def preprocess_numpy(img):
    resize = transforms.Resize((224, 224))   #must same as here
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    img.unsqueeze_(0)
    return img

# LOADING ONNX FILE FOR INFERENCE 
onnx_session= onnxrt.InferenceSession("mtailor-resnet.onnx")

# GETTING THE 2 TEST IMAGES
img1 = Image.open("mtailor_mlops_assessment/n01440764_tench.jpeg")
img1 = preprocess_numpy(img1)
img2 = Image.open("mtailor_mlops_assessment/n01667114_mud_turtle.JPEG")
img2 = preprocess_numpy(img2)

# RUNNING AND ASSERTING ON FIRST IMAGE
onnx_inputs= {
    onnx_session.get_inputs()[0].name: to_numpy(img1)
}
onnx_output = onnx_session.run(None, onnx_inputs)
img_label = onnx_output[0]
assert np.argmax(img_label) == 0

# RUNNING AND ASSERTING ON SECOND IMAGE
onnx_inputs= {
    onnx_session.get_inputs()[0].name: to_numpy(img2)
}
onnx_output = onnx_session.run(None, onnx_inputs)
img_label = onnx_output[0]
assert np.argmax(img_label) == 35
