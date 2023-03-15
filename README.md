# mtailor-task

This is a trail for MTailor task, with a lot of bad habits.

## How to run
The first part is to transform and run model on ONNX Format.
I did this command to trnasform to ONNX:
```shell
python3 mtailor_mlops_assessment/convert_to_onnx.py -f mtailor_mlops_assessment/pytorch_model.py -c resnet18-f37072fd.pth
```
which require few parameters:
1. the main model python file.
2. the parameter file after training.

And that shall build a new ONNX file that can be used. After that, you can run the following command to check if the model is running and ouput is valid:
```shell
python3 mtailor_mlops_assessment/test_onnx.py -m mtailor_mlops_assessment/resnet50.onnx
```
it takes the ONNX file and then validate if it is running and checks the output of the 2 given images.

# Becareful ⚠ 
I assumed that the images are in the same directory of the files, which is very bad habit, but for the sake of time ⌚ I had to do that.
I also assumed that the model parameters file has a static name, which cost me a lot while tracing that.
