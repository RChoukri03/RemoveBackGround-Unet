from keras_unet_collection import models
import tensorflow as tf
import onnxmltools
# inputShape=(320,320,3)
input_shape=(320, 320, 3)


def h5toOnnx(model, outputPath):
    # Convert the H5 model to ONNX format
    onnx_model = onnxmltools.convert.convert_keras(model)
    onnxmltools.utils.save_model(onnx_model, outputPath)
    return outputPath


modelConfig = {
        'input_size': input_shape,
        'n_labels': 1,
        'filter_num_down': [64, 64, 64, 64],
        'filter_num_up': [64, 64, 64, 64],
        'filter_mid_num_down': [16, 16, 16, 16],
        'filter_mid_num_up': [16, 16, 16, 16],
        'filter_4f_num': [32, 32],
        'filter_4f_mid_num': [16, 16],
        'activation': 'ReLU',
        'output_activation': 'Sigmoid',
        'batch_norm': True,
        'pool': 'max',
        'unpool': 'bilinear',
        'deep_supervision': True,
        'name': 'u2netKit'
    }

# Vous pouvez ensuite utiliser ce dictionnaire pour configurer votre modèle comme suit :
model = models.u2net_2d(**modelConfig)
# Load the Keras H5 model
bestModelPath = 'weights/u2netKitBest.h5' 

to = model.load_weights(bestModelPath)
h5toOnnx(model=model, outputPath='u2net_kit.onnx')
# Convert the H5 model to ONNX format
# onnx_model_path = 'tst.onnx'
# onnx_model = onnxmltools.convert.convert_keras(model)
# onnxmltools.utils.save_model(onnx_model, onnx_model_path)

#Run this command to convert a TFlite model to ONNX - We don't use it because it degrade performance of NN.

'''TFLITE TO ONNX /   # python -m tf2onnx.convert --tflite path-to-tflite-model --output path-to-onnx-modelx   
'''
