FROM python:latest

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# 🍌: Add your custom app code, init() and inference()
ADD . .

EXPOSE 8000

RUN python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3 -m pip install six numpy Pillow onnx

RUN wget https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth
RUN python3 convert_to_onnx.py -f pytorch_model.py -c pytorch_model_weights.pth
RUN python3 test_onnx.py -m pytorch_model_weights.onnx
