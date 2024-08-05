import tensorflow as tf
from keras_unet_collection import models
import cv2
import numpy as np
from PIL import Image
import time

# Charger le modèle U2NET non quantifié
#model lourd
inputShape=(320,320,3)
model = models.u2net_2d(inputShape, n_labels=1,
                       filter_num_down=[32,64, 128, 256, 512], filter_num_up=[32,64,128,256,512],
                       filter_mid_num_down=[32, 64, 128, 256, 512], filter_mid_num_up=[32, 64, 128, 256, 512],
                       filter_4f_num=[256, 256], filter_4f_mid_num=[128,128],
                       activation='ReLU', output_activation='Sigmoid',
                       batch_norm=True, pool='max', unpool='bilinear', deep_supervision=False, name='u2net')

# Entraîner ou charger les poids de votre modèle U2NET
resume = 'weights/u2net_chkLourdbest.h5'
model.load_weights(resume)
model.summary()
# Convertir le modèle en modèle quantifié
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Appliquer l'optimisation par défaut (y compris la quantification)
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
quantized_tflite_model = converter.convert()

# Sauvegarder le modèle quantifié dans un fichier
with open('u2net_latency.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

#Run this command to convert a TFlite model to ONNX - We don't use it because it degrade performance of NN.

'''
# python -m tf2onnx.convert --tflite path-to-tflite-model --output path-to-onnx-modelx   
'''


