FROM python3:latest

# 🍌: Add your custom app code, init() and inference()
ADD . .

EXPOSE 8000

RUN pip install onnx

RUN wget https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth
RUN python3 mtailor_mlops_assessment/convert_to_onnx.py -f mtailor_mlops_assessment/pytorch_model.py -c resnet18-f37072fd.pth
RUN python3 mtailor_mlops_assessment/test_onnx.py -m mtailor_mlops_assessment/resnet50.onnx