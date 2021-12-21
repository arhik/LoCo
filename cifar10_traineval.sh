#!/bin/sh

echo "Training the Greedy InfoMax Model on vision data (imagenet)"
python -m GreedyInfoMax.vision.main_vision --grayscale --download_dataset  --save_dir vision_experiment_cifar --dataset CIFAR10 --num_epochs 100 --batch_size 32
echo "Testing the Greedy InfoMax Model for image classification"
python -m GreedyInfoMax.vision.downstream_classification --grayscale --model_path ./logs/vision_experiment_cifar --model_num 99
