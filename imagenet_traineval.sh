#!/bin/sh

echo "Training the Greedy InfoMax Model on vision data (imagenet)"
python -m GreedyInfoMax.vision.main_vision --grayscale --save_dir vision_experiment --dataset imagenetmini --num_epochs 100
echo "Testing the Greedy InfoMax Model for image classification"
python -m GreedyInfoMax.vision.downstream_classification --grayscale --dataset imagenetmini --model_path ./logs/vision_experiment --model_num 100
