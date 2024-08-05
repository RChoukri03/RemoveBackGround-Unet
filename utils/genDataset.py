import shutil
import numpy as np
import json
import os
import random
import string
import logging
from datetime import datetime
from PIL import Image, ImageDraw, ImageFilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

predictImages = 100
basePath = 'Segmentation'
basePathBatches = basePath
basePathDataset = f'{basePath}/Datasets'

# Generate dataset name
now = datetime.now()
dsGenName = now.strftime('%d%m%Y%H%M%S') + ''.join((random.choice(string.ascii_letters) for x in range(5)))
fullPathDatasetName = f'{basePathDataset}/{dsGenName}'

# Create directories
logging.info(f'Creating path {fullPathDatasetName}/src/')
os.makedirs(f'./{fullPathDatasetName}/src/', exist_ok=True)
os.makedirs(f'./{fullPathDatasetName}/mask/', exist_ok=True)
os.makedirs(f'./{fullPathDatasetName}/predict/', exist_ok=True)

treatedImgs = []

# Loop over every batch group
for batchGroup in os.listdir(basePathBatches):
    fullPathBatchGroup = f'{basePathBatches}/{batchGroup}'

    # Skip 'crop.txt' and other files in the root of each batch group folder
    for batch in (file for file in os.listdir(fullPathBatchGroup) if file != 'crop.txt'):
        fullPathBatch = f'{fullPathBatchGroup}/{batch}'
        cropDict = None

        # Open crop file if it exists
        try:
            with open(f'{fullPathBatchGroup}/crop.txt') as f:
                cropDict = json.loads(f.read())
        except FileNotFoundError:
            pass

        # Process each batch
        for batch in os.listdir(fullPathBatch):
            batchPath = f'{fullPathBatch}/{batch}'

            # Try to find json file
            logging.info(f'Trying: {batchPath}')
            jsonFiles = [f for f in os.listdir(batchPath) if f.endswith('.json')]

            # If the json file exists
            if len(jsonFiles) == 1:
                logging.info(f'Found: {jsonFiles[0]}')
                annotationFilePath = f'{batchPath}/{jsonFiles[0]}'
                with open(annotationFilePath) as f:
                    logging.info(f'Opened: {batchPath}')
                    data = json.loads(f.read())
                    imgs = data['images']

                    # Sort annotations to ensure 'hole' annotations come after 'instrument'
                    sortedAnnotations = sorted(data['annotations'], key=lambda item: item['category_id'])

                    # Loop over annotations to build the masks
                    for annot in sortedAnnotations:
                        categoryId = annot['category_id'] - 1
                        try:
                            cat = data['categories'][categoryId]['name']
                        except IndexError:
                            cat = 'unsupported index'
                            logging.error(f'Unsupported category index {categoryId}')

                        if cat == 'kit' or cat == 'hole':
                            imgId = annot['image_id']
                            y = annot['segmentation'][0][1::2]
                            x = annot['segmentation'][0][::2]

                            img = imgs[imgId - 1]
                            imgName = img['file_name']
                            pngImgName = imgName.split('.')[0] + '.png'
                            imgWidth = img['width']
                            imgHeight = img['height']

                            # Load the source image
                            imgSrc = Image.open(f'{batchPath}/{imgName}')

                            # Create or reuse the mask
                            msk = None
                            reuse = False
                            if imgName in treatedImgs:
                                logging.info(f'Image {pngImgName} already used, reopening to insert new mask')
                                msk = Image.open(f'{fullPathDatasetName}/mask/{pngImgName}')
                                reuse = True
                            else:
                                msk = Image.new('L', (imgWidth, imgHeight))

                            # Crop the canvas
                            offsetX = 0
                            offsetY = 0
                            cropDict = {
                                "fL": [150, 270, 0, 170],
                                "fR": [0, 270, 180, 170],
                                "fM": [100, 270, 100, 170],
                                "fN": [0, 0, 0, 0]
                            }

                            w, h = imgSrc.size
                            imgPos = imgName[0:2]
                            margins = cropDict.get(imgPos, [0, 0, 0, 0])
                            marginLeft, marginTop, marginRight, marginBottom = margins
                            offsetX = marginLeft
                            offsetY = marginTop
                            cropBox = (marginLeft, marginTop, w - marginRight, h - marginBottom)
                            imgSrc = imgSrc.crop(cropBox)

                            if not reuse:
                                msk = msk.crop(cropBox)

                            # Draw the mask
                            msk1 = ImageDraw.Draw(msk)
                            xOffseted = [val - offsetX for val in x]
                            yOffseted = [val - offsetY for val in y]
                            xy = [(i, j) for i, j in zip(xOffseted, yOffseted)]
                            fillColor = 'white' if cat == 'kit' else 'black'
                            msk1.polygon(xy, fill=fillColor, width=0)
                            msk.filter(ImageFilter.SMOOTH)

                            # Save the mask and source image
                            try:
                                msk.save(f'{fullPathDatasetName}/mask/{pngImgName}', compress_level=0)
                                imgSrc.save(f'{fullPathDatasetName}/src/{imgName}')
                                treatedImgs.append(imgName)
                            except FileNotFoundError:
                                logging.error('Could not write on target')
                                exit(-1)
                        else:
                            logging.warning(f'Unsupported category: {cat}')
            elif len(jsonFiles) == 0:
                # Transfer some images for prediction if no json file exists
                cropDict = {
                    "fL": [150, 270, 0, 170],
                    "fR": [0, 270, 180, 170],
                    "fM": [100, 270, 100, 170],
                    "fN": [0, 0, 0, 0]
                }
                logging.info('To Predict')
                if predictImages >= 0:
                    for imgName in os.listdir(batchPath):
                        logging.info(imgName)
                        if imgName.split('.')[1] == 'jpeg':
                            img = Image.open(f'{batchPath}/{imgName}')
                            w, h = img.size
                            imgPos = imgName[0:2]
                            margins = cropDict.get(imgPos, [0, 0, 0, 0])
                            marginLeft, marginTop, marginRight, marginBottom = margins
                            cropBox = (marginLeft, marginTop, w - marginRight, h - marginBottom)
                            img = img.crop(cropBox)
                            img.save(f'{fullPathDatasetName}/predict/{imgName}')
                            predictImages -= 1
                        else:
                            logging.warning('Unknown format')
            else:
                logging.error('Inconsistent number of annotation files')
