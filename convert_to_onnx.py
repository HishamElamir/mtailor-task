# FILES IMPORTS
import torch
from PIL import Image
from pytorch_model import Classifier, BasicBlock

# INIT MODEL
mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
mtailor.load_state_dict(torch.load("pytorch_model_weights.pth"))
mtailor.eval()

# LOADING TEST IMAGE
img = Image.open("n01667114_mud_turtle.JPEG")
inp = mtailor.preprocess_numpy(img).unsqueeze(0) 
res = mtailor.forward(inp)

# INPUT/OUTPUT PARAMS
input_names = ["actual_input"]
output_names = ["output"]

# EXPORTING THE MODEL
torch.onnx.export(
    mtailor,                    # MODEL
    inp,                        # INPUT SAMPLE IMAGE
    "mtailor-resnet.onnx",      # ONNX FILE OUTPUT
    verbose=False,
    input_names=input_names,    # INPUT NAME
    output_names=output_names,  # OUTPUT NAME
    export_params=True
)
