# U2Net-Based Image Segmentation Project

## Project Overview

This project focuses on image segmentation using the U2Net architecture. The main goal is to develop a robust model capable of accurately segmenting specific objects within images. The dataset used for training and testing includes annotated images with various objects. The project is structured to handle training, inference, and pruning of the U2Net model.

## Features

- **Training**: Train the U2Net model with a specified dataset.
- **Inference**: Perform inference to generate segmentation masks for new images.
- **Pruning**: Prune the trained model to optimize performance.
- **Data Augmentation**: Utilize data augmentation techniques to enhance model generalization.
- **ClearML Integration**: Leverage ClearML for experiment management, model tracking, and artifact management.

## ClearML Integration

ClearML is used in this project for several reasons:

1. **Experiment Management**: ClearML helps in managing and tracking various training experiments, making it easier to compare results and reproduce experiments.
2. **Model Tracking**: It provides a centralized repository for tracking model versions, hyperparameters, and performance metrics.
3. **Artifact Management**: ClearML allows for seamless uploading and downloading of model artifacts, ensuring that models and related files are organized and easily accessible.
4. **Visualization**: It offers powerful visualization tools for monitoring training progress, including loss curves, accuracy plots, and more.

## Setup Environment

To set up the environment, you need to build the Docker image:

```bash
docker build -t rmbg:tag .

```

## Training
To train the U2Net model, use the following command:

```bash
python main.py --task train --dataset Name_Of_Dataset --BatchSize 12 --epochs 500 --learning_rate 0.001 --resume Path_To_Pretrained_Weights
```