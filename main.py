import os
import argparse
import pathlib
import logging
import model.u2net as u2net

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
datasets_dir = pathlib.Path(__file__).absolute().parent.joinpath('images/')

# CLI arguments
parser = argparse.ArgumentParser(description='U2 Net')
parser.add_argument('--dataset', default=None, type=str, help='Dataset to use for training or pruning')
parser.add_argument('--src', default=None, type=str, help='Source directory or file for inference')
parser.add_argument('--resume', default=None, type=str, help='Resume training or pruning from a checkpoint')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training or pruning')
parser.add_argument('--task', default='train', type=str, choices=['train', 'predict', 'infer', 'prune'], help='Task to perform: train, predict, infer, prune')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training or pruning')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
args = parser.parse_args()

# Assign CLI arguments to variables
resume = args.resume
batch_size = args.batch_size
src = args.src
dataset = args.dataset
epochs = args.epochs
task = args.task
learning_rate = 0.001

# Validate task argument
allowed_tasks = ['train', 'predict', 'infer', 'prune']
if not task:
    logging.error('No task specified ... abort')
    exit(-1)
elif task not in allowed_tasks:
    logging.error(f'Task ({task}) does not exist ... abort')
    exit(-1)

# Handle dataset requirement for training and pruning tasks
if task in ['train', 'prune']:
    if not dataset:
        logging.error('No dataset provided ... here are the available datasets:')
        for d in os.listdir(datasets_dir):
            logging.info(f'> {d}')
        exit(-1)
    else:
        dataset_dir = datasets_dir.joinpath(dataset)

# Execute the specified task
if task == 'train':
    # Training
    logging.info(f'Starting training on dataset: {dataset} for {epochs} epochs with batch size {batch_size}')
    u2net.train(dataset_dir, batch_size, epochs, learning_rate, resume)

elif task in ['infer', 'predict']:
    # Inference/Predict
    if not src:
        logging.error('Inference requires a directory or file provided with --src option ... abort now')
        exit(-1)
    logging.info(f'Starting inference on source: {src}')
    u2net.infer(src, resume)

elif task == 'prune':
    # Pruning
    if not resume:
        logging.error('You can only prune networks that have been trained ... no resume checkpoint provided with --resume option ... abort now')
        exit(-1)
    logging.info(f'Starting pruning on dataset: {dataset} for {epochs} epochs with batch size {batch_size}')
    u2net.prune(dataset_dir, batch_size, epochs, resume)
