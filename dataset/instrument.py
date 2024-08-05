

import os
from cv2 import imwrite
import keras
import numpy as np
import random
import cv2 as cv
import time

# Function for Data Augmentation > allow us to train on multiple resolutions
def random_crop(image, msk):
         
        mask = cv.cvtColor(msk, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            bounding_boxes = [cv.boundingRect(contour) for contour in contours]
            func = random.choice([max,min])
            x, y, w, h = func(bounding_boxes, key=lambda item: item[2]*item[3]) 

            crop_size = (max(w, h) + random.randint(0, min(w,h)+random.randint(0,1500)), max(w, h) + random.randint(0, min(w,h)+random.randint(0,1500)))

            start_x = random.randint(max(0, x - (crop_size[0]-w)), x)
            start_y = random.randint(max(0, y - (crop_size[1]-h)), y)

            image_crop = image[start_y:start_y+crop_size[1], start_x:start_x+crop_size[0]]
            mask_crop = msk[start_y:start_y+crop_size[1], start_x:start_x+crop_size[0]]
        else:
            image_crop = image
            mask_crop = msk

        return image_crop, mask_crop

class InstrumentDataset(keras.utils.Sequence):

    def __init__(self, datasetDir, batchSize, imgSize, shuffleSeed = 1337, validSplit = None, valid = False):

        self.datasetDir = datasetDir

        self.batchSize = batchSize
        self.imgSize = imgSize

        self.suffleSeed = shuffleSeed
        self.validSplit = validSplit
        self.isValid = valid
        
        self.augmenter = None

        self.initPaths(display=True)

    def setAugmenter(self, albumentations):
        self.augmenter = albumentations

    def initPaths(self, display=False):
        inputDir = f'{self.datasetDir}/src'
        maskDir = f'{self.datasetDir}/mask'

        # Get Src Paths
        self.inputImgPaths = sorted(
            [
                os.path.join(inputDir, fname)
                for fname in os.listdir(inputDir)
                if fname.endswith('.jpeg')
            ]
        )

        # Get Msk Paths
        self.maskImgPaths = sorted(
            [
                os.path.join(maskDir, fname)
                for fname in os.listdir(maskDir)
                if fname.endswith('.png') and not fname.startswith(".")
            ]
        )

        # suffle the data before using it
        random.Random(1337).shuffle(self.inputImgPaths)
        random.Random(1337).shuffle(self.maskImgPaths)

        # Make val split 
        if self.validSplit is not None:
            cutIdx = int(len(self.inputImgPaths) - len(self.inputImgPaths) * self.validSplit)
            if not self.isValid:
                self.inputImgPaths = self.inputImgPaths[:cutIdx]
                self.maskImgPaths = self.maskImgPaths[:cutIdx]
                print("Number of samples in train dataset:", len(self.inputImgPaths))
            else:
                self.inputImgPaths = self.inputImgPaths[cutIdx:]
                self.maskImgPaths = self.maskImgPaths[cutIdx:]
                print("Number of samples in validation dataset:", len(self.inputImgPaths))
        else:
            print("Number of samples in train dataset:", len(self.inputImgPaths))

        if display:
            # Show small display of input data
            for inputPath, maskPath in zip(self.inputImgPaths[:10], self.maskImgPaths[:10]):
                print(f'{inputPath} | {maskPath}')

    @property
    def numImages(self):
        return len(self.maskImgPaths)

    def __len__(self):
        return len(self.maskImgPaths) // self.batchSize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchSize
        batch_inputImgPaths = self.inputImgPaths[i : i + self.batchSize]
        batch_maskImgPaths = self.maskImgPaths[i : i + self.batchSize]
        X = np.zeros((self.batchSize,) + self.imgSize + (3,), dtype="float32")
        y = np.zeros((self.batchSize,) + self.imgSize + (1,), dtype="float32")
        for j, (pathImg, pathMask) in enumerate(zip(batch_inputImgPaths,batch_maskImgPaths)):
            # Img
            msk = cv.imread(pathMask)
            img = cv.imread(pathImg)
            height,width,_=img.shape
            random.seed(time.time())
            rand = random.randint(0,9999999)
            if rand % 7 !=0:
                img, msk= random_crop(img, msk)
            # Resize
            img = cv.resize(img,self.imgSize)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Msk
            
            msk = cv.resize(msk, self.imgSize)
            msk = cv.cvtColor(msk, cv.COLOR_BGR2GRAY)[:,:,None]

            # Augmentations
            if self.augmenter is not None and not self.isValid:
                transformed = self.augmenter(image=img,mask=msk)
                img = transformed['image']
                msk = transformed['mask']

            # Write to disk for debuging albumentation purpose
            # imwrite(f'tmp/{i}_{j}.png', cv.cvtColor(img, cv.COLOR_RGB2BGR))
            # imwrite(f'tmp/{i}_{j}_mask.png', msk)

            # inject img and mask
            X[j] = img / 255.0
            y[j] = msk / 255.0

        return X, y

