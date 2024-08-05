

from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import time


def loadAndReshapeImg(imageName, inputShape):
    # Ouvrir l'image en tant qu'image PIL RGB pour plus de commodité
    imageData = Image.open(imageName)
    size = np.array(imageData).shape[:2]
    imageData = imageData.resize((inputShape[0], inputShape[1]))  # Redimensionner à la forme d'entrée
    imageData = np.array(imageData) / 255.  # Normaliser les valeurs des pixels entre 0 et 1
    return np.expand_dims(imageData, axis=0), size

def resizeAndWriteImg(imageData, imageName, origSize):
    # Redimensionner à la taille d'origine
    imageData = cv2.resize(imageData, (origSize[1], origSize[0]))  # Redimensionner à (largeur, hauteur)
    imageData = np.tile(np.expand_dims(imageData, axis=2), [1, 1, 3])
    imageData = cv2.cvtColor(imageData.astype('float32'), cv2.COLOR_RGB2BGR) * 255.
    
    # Enregistrer l'image sur le disque
    outputImageName = imageName.split('/')[-1].split('.')[0] + '_masked.jpeg'
    cv2.imwrite(f'./out/{outputImageName}', imageData)


# Charger le modèle quantifié
interpreter = tf.lite.Interpreter(model_path="weights/u2net_quantized.tflite")
interpreter.allocate_tensors()



# Charger les poids du modèle quantifié
input_tensor_details = interpreter.get_input_details()
output_tensor_details = interpreter.get_output_details()
input_tensor_index = input_tensor_details[0]['index']
output_tensor_index = output_tensor_details[0]['index']





src = 'images/Datasets/RMBGTEST/middle1.jpeg'
inputShape = (320, 320, 3)
input_image, origSize = loadAndReshapeImg(src, inputShape)
# Convertir les valeurs de l'image d'entrée en FLOAT32
input_image = input_image.astype(np.float32)
print('done1')
# Effectuer l'inférence
interpreter.set_tensor(input_tensor_index, input_image)
print('done2')
start_time = time.time()
interpreter.invoke()
end_time = time.time()
execution_time = end_time - start_time

print("Temps d'inference':", execution_time, "secondes")
maskData = interpreter.get_tensor(output_tensor_index)
print('done3')
# Utiliser les résultats de l'inférence
resizeAndWriteImg(maskData[0], src, origSize)
