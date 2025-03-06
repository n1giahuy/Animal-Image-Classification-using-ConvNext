# Animal Image Classification using ConvNext

<div align="center">
  <img src="https://cdn.dribbble.com/users/310241/screenshots/10620235/_________-______1-dribbble.gif" width="600" height="300" alt="Project Introduction">
</div>

## Introduction
This implementation uses transfer learning with the ConvNeXt model, a modern CNN architecture introduced in ["A ConvNet for the 2020s"](https://arxiv.org/abs/2201.03545) paper. I experimented with three different ConvNext model variants(tiny, small, and base) to compare their performance on image classification, then visualized their metrics using Tensorboard. Finally, I tested the trained models with images downloaded from the Internet.

## Data Collection
For this project, I used the data available on [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10) .It contains about 28K medium quality animal images belonging to 10 categories: dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, and elephant.

## Trained Model
You can download the best trained model from [here](https://drive.google.com/drive/u/1/folders/10_HcQGuVygDJ55h9K5776lkalL4gJTiU) Run it with mode_size="base". 
