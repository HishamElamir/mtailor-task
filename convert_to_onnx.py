import torch
from PIL import Image

from pytorch_model import Classifier, BasicBlock


mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
mtailor.load_state_dict(torch.load("pytorch_model_weights.pth"))
mtailor.eval()

img = Image.open("n01667114_mud_turtle.JPEG")
inp = mtailor.preprocess_numpy(img).unsqueeze(0) 
res = mtailor.forward(inp)

input_names = ["actual_input"]
output_names = ["output"]

torch.onnx.export(
    mtailor,
    inp,
    "mtailor-resnet.onnx",
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    export_params=True
)
