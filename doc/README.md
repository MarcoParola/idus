# **Documentation**

The project is composed of the following modules, more details are below:

- [Main scripts for training and test models](#main-scripts-for-training-and-test-models)
- [Deep learning architecture implementation](#deep-learning-models)
- [Configuration handling](#configuration-handling)
- [Additional utility scripts](#additional-utility-scripts)

## **Installation**

To install the project, simply clone the repository and get the necessary dependencies. Then, create a new project on [Weights & Biases](https://wandb.ai/site). Log in and paste your API key when prompted.
```sh
# clone repo
git clone https://github.com/MarcoParola/idus.git
cd idus
mkdir models data

# Create virtual environment and install dependencies 
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt 

# Weights&Biases login 
wandb login 
```

## **Usage**

To perform a training run by setting `model` parameter that can assume the following value `detr`, `yolos`, `yolo`.
```sh
python train.py model=detr
```

 

To run inference on test set to compute some metrics, specify the weight model path by setting `weight` parameter (I ususally download it from wandb and I copy it in `checkpoint` folder).
```sh
python test.py model=detr weight=checkpoint/best.pt
```

## **Training params**
Training hyperparams can be edited in the [config file](./config/config.yaml) or ovewrite by shell


## Main scripts for training and test models

All the experiments consist of train and test transformer architectures. You can use `train.py` and `test.py`, respectively. Please, note model weights file should be properly stored.

```bash
python train.py
python test.py weight=path/to/model-weight
```


## Deep learning models

## Configuration handling
The configuration managed with [Hydra](https://hydra.cc/). Every aspect of the configuration is located in `config/` folder. The file containing all the configuration is `config.yaml`.

## Additional utility scripts

In the `scripts/` folder, there are all independent files not involved in the `pytorch` workflow for data preparation and visualization.
