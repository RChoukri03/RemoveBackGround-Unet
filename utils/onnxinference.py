import cv2
import numpy as np
import onnxruntime as rt
import time 
def loadAndReshapeImg(imageName, inputShape):
    imageData = cv2.imread(imageName)  # Load image using OpenCV
    size = imageData.shape[:2]
    imageData = cv2.resize(imageData, (inputShape[1], inputShape[0]))  # Resize to input shape
    imageData = np.array(imageData) / 255.  # Normalize pixel values between 0 and 1
    return np.expand_dims(imageData, axis=0), size

def resizeAndWriteImg(imageData, imageName, origSize):
    imageData = cv2.resize(imageData, (origSize[1], origSize[0]))  # Resize to original size
    imageData = np.tile(np.expand_dims(imageData, axis=2), [1, 1, 3])
    imageData = cv2.cvtColor(imageData.astype('float32'), cv2.COLOR_RGB2BGR) * 255.
    
    # Save the image
    outputImageName = imageName.split('/')[-1].split('.')[0] + '_masked.jpeg'
    cv2.imwrite(f'./out/{outputImageName}', imageData)
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = -1

sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL #ORT_PARALLEL 
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

# Charger le modèle ONNX avec les options de session
onnx_model_path = './weights/BigOne.onnx'

session = rt.InferenceSession(onnx_model_path , sess_options)

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load and preprocess the input image
src = 'images/Datasets/RMBGTEST/middle1.jpeg'
inputShape = (320, 320, 3)
input_image, origSize = loadAndReshapeImg(src, inputShape)
input_image = input_image.astype(np.float32)

# Perform inference
start_time = time.time()
output = session.run([output_name], {input_name: input_image})
end_time = time.time()
execution_time = end_time - start_time
print("Inference time:", execution_time, "seconds")

# Process the output
maskData = output[0]

# Use the inference results
resizeAndWriteImg(maskData[0], src, origSize)
