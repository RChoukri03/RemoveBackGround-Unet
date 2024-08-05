import os
import cv2 as cv
from PIL import Image
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import albumentations as A
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_unet_collection import models, losses
from dataset.instrument import InstrumentDataset
from utils.clearml import ClearMLManager
from utils.h5toonnx import h5toOnnx
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU setup
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        logging.info(f"GPU found: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    logging.error(f"GPU setup error: {e}")

# ClearML manager initialization
clearMLManager = ClearMLManager(projectName='Kit_Segmentation', taskName='Train')

# Helper function to log data
def log(file_path, message):
    logging.info(message)
    with open(file_path, "a") as f:
        f.write(message + '\n')

# Model shapes
defaultInputShape = (640, 640, 3)
defaultOutputShape = (640, 640, 1)

# Define the U2net model
def getU2netModel(input_shape=(640, 640, 3), learning_rate=0.001):
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

    model = models.u2net_2d(**modelConfig)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', losses.dice_coef])
    
    model.summary(print_fn=lambda x: clearMLManager.getLogger().report_text(x))
    return model, modelConfig

# Training function
def train(datasetDir, batchSize=16, epochs=1000, learningRate=0.001, resume=None, logFile='./my_log.txt'):
    clearMLManager.connectConfig({
        'dataset_dir': datasetDir,
        'batch_size': batchSize,
        'epochs': epochs,
        'learning_rate': learningRate,
    }, 'TrainingConfiguration')

    log(logFile, f'Training begins at: {datetime.now()}')

    # Data augmentation setup
    trainAlbs = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomContrast(limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightness(limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.9),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=45, border_mode=cv.BORDER_REFLECT_101, p=0.8, interpolation=2),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAASharpen(p=0.2),
        A.IAAEmboss(p=0.2),
        A.RandomShadow(p=0.2),
        A.ChannelShuffle(p=0.02)
    ])

    # Dataset preparation
    trainInstDsGen = InstrumentDataset(datasetDir, batchSize=batchSize, imgSize=defaultInputShape[:-1], validSplit=0.1, valid=False)
    valInstDsGen = InstrumentDataset(datasetDir, batchSize=batchSize, imgSize=defaultInputShape[:-1], validSplit=0.1, valid=True)
    trainInstDsGen.setAugmenter(trainAlbs)

    # Model setup
    model, modelConfig = getU2netModel(input_shape=defaultInputShape, learning_rate=learningRate)
    if resume:
        model.load_weights(resume)

    # TensorBoard and ClearML logging
    log_dir = f'./logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch')

    bestModelPath = 'u2netKitBest.h5'
    callbacks = [
        ModelCheckpoint(bestModelPath, save_best_only=True, save_weights_only=True, verbose=1),
        tensorboard_callback,
        EarlyStopping(patience=30)
    ]

    # Training process
    model.fit(trainInstDsGen, batch_size=batchSize, epochs=epochs, validation_data=valInstDsGen, callbacks=callbacks)

    # Save final results
    timeMarker = datetime.now().strftime('%H%M%S-%d%m%y')
    targetPath = f'weights/u2net-{epochs}e-{timeMarker}.h5'
    log(logFile, f'Saving final results to {targetPath}')
    if not os.path.exists('weights'):
        os.makedirs('weights')
    model.save_weights(targetPath)

    # Convert model to ONNX and upload to ClearML
    outputPath = h5toOnnx(model=model, outputPath='finalKitSegment.onnx')
    clearMLManager.uploadArtifacts(name='finalSegmentKitModel', object=outputPath, metadata=modelConfig)

    model.load_weights(bestModelPath)
    outputPath = h5toOnnx(model=model, outputPath='bestKitSegment.onnx')
    clearMLManager.uploadArtifacts(name='bestSegmentKitModel', object=outputPath, metadata=modelConfig)

    # Closing ClearML task
    clearMLManager.close()

# Inference function
def infer(src, weights, batchSize=9, logFile='./my_log.txt'):
    def loadAndReshapeImg(imageName):
        log(logFile, f'Loading image from {imageName}')
        imageData = Image.open(imageName).convert('RGB')
        return cv.resize(np.array(imageData), defaultInputShape[:-1]) / 255., imageData.size

    def resizeAndWriteImg(imageData, imageName, origSize):
        imageData = cv.resize(imageData, origSize)
        imageData = np.tile(np.expand_dims(imageData, axis=2), [1, 1, 3])
        imageData = cv.cvtColor(imageData.astype('float32'), cv.COLOR_RGB2BGR) * 255.
        outputImageName = imageName.split('/')[-1].split('.')[0] + '_masked.jpeg'
        log(logFile, f'Outputting image to out/{outputImageName}')
        cv.imwrite(f'./out/{outputImageName}', imageData)

    log(logFile, f'Inference begins at: {datetime.now()}')
    model, _ = getU2netModel()

    if weights:
        log(logFile, f'Loading weights from {weights}')
        model.load_weights(weights)
    else:
        logging.error('No model weight provided... exit')
        exit(-1)

    if os.path.isfile(src):
        imageData, origSize = loadAndReshapeImg(src)
        imageData = np.expand_dims(np.array(imageData), 0)
        maskData = model.predict(imageData)[0]
        resizeAndWriteImg(maskData, src, origSize)
    elif os.path.isdir(src):
        for batchFiles in batch(os.listdir(src), batchSize):
            imageBatch = []
            nameBatch = []
            sizeBatch = []
            for file in batchFiles:
                imageName = f'{src}/{file}'
                imageData, origSize = loadAndReshapeImg(imageName)
                imageBatch.append(imageData)
                nameBatch.append(imageName)
                sizeBatch.append(origSize)

            if len(imageBatch) != batchSize:
                missing = batchSize - len(imageBatch)
                for i in range(missing):
                    imageBatch.append(np.zeros(defaultInputShape))

            imageBatch = np.array(imageBatch)
            maskBatch = model.predict(imageBatch, batch_size=len(imageBatch))

            maskBatch = maskBatch[:len(imageBatch), :, :]

            for maskData, imageName, origSize in zip(maskBatch, nameBatch, sizeBatch):
                resizeAndWriteImg(maskData, imageName, origSize)
    else:
        logging.error(f'No source image provided or invalid ({src})... exit')
        exit(-1)

# Pruning function
def prune(datasetDir=None, batchSize=9, epochs=50, resume=None, logFile='./my_log.txt'):
    log(logFile, f'Pruning begins at: {datetime.now()}')

    trainInstDsGen = InstrumentDataset(datasetDir, batchSize=batchSize, imgSize=defaultInputShape[:-1], validSplit=0.005, valid=False)
    valInstDsGen = InstrumentDataset(datasetDir, batchSize=batchSize, imgSize=defaultInputShape[:-1], validSplit=0.4, valid=True)

    model, optimizer = getU2netModel()

    if resume:
        log(logFile, f'Loading weights from {resume}')
        model.load_weights(resume)

    n = trainInstDsGen.numImages
    end_step = np.ceil(n / batchSize).astype(np.int32) * epochs

    pruningArgs = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.45, final_sparsity=0.75, begin_step=0, end_step=end_step)
    }

    modelForPruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruningArgs)

    modelForPruning.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', losses.dice_coef])
    modelForPruning.summary(print_fn=lambda x: clearMLManager.getLogger().report_text(x))

    log_dir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
        EarlyStopping(patience=30)
    ]

    modelForPruning.fit(trainInstDsGen, batch_size=batchSize, epochs=epochs, validation_data=valInstDsGen, callbacks=callbacks)

    modelForExport = tfmot.sparsity.keras.strip_pruning(modelForPruning)
    modelForExport.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', losses.dice_coef])
    modelForExport.fit(trainInstDsGen, batch_size=batchSize, epochs=epochs, validation_data=valInstDsGen)

    prunedWeightsPath = resume.split('.')[0] + '_pruned.h5'
    modelForExport.save_weights(prunedWeightsPath)
    log(logFile, f'Saved pruned weights to {prunedWeightsPath}')

if __name__ == "__main__":
    logging.error("Don't run this file directly.")
