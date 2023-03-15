FROM python:latest

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# 🍌: Add your custom app code, init() and inference()
ADD . .

EXPOSE 8000

RUN python3 -m pip install numpy Pillow torch torchvision onnx

RUN wget https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth
RUN python3 mtailor_mlops_assessment/convert_to_onnx.py -f mtailor_mlops_assessment/pytorch_model.py -c resnet18-f37072fd.pth
RUN python3 mtailor_mlops_assessment/test_onnx.py -m mtailor_mlops_assessment/resnet50.onnx
