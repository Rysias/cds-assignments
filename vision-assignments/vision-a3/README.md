# Assignment 3 - Transfer learning + CNN classification
This assignment implements transfer learning using the VGG16 neural network for effectively classifying images. This is (hopefully) an improvement over the simpler used in assignment 2. In implementing this project, I have relied on [this tensorflow tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning) about transfer learning as well as [this one](https://www.tensorflow.org/tutorials/images/cnn) about convolutional neural networks. 

However, instead of blindly following the tutorials, I have updated them using the SOLID-principles, which I will further describe in the next section. I have also used GitHub Copilot, a code completition tool, which I believe is a [game changer for coding](https://medium.com/codex/github-copilot-is-a-game-changer-cd0a2bbe6de8)
## Table of Content
- [Assignment Description](#assignment-description)
    * [Personal learning goals](#personal-learning-goals)
- [Methods and design](#methods-and-design)
    * [Software design](#software-design)
- [Usage](#usage)
    * [Setting up](#setting-up)
    * [Using the script(s)](#using-the-scripts)
- [Results and Discussion](#results-and-discussion)
    * [Results](#results)
    * [Discussion](#discussion)

## Assignment Description
In this assignment, you are still going to work with the CIFAR10 dataset. However, this time, you are going to make build a classifier using transfer learning with a pretrained CNN like VGG16 for feature extraction. 

Your ```.py``` script should minimally do the following:

- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report

**Additional bonus tasks completed**:

- Use ```argparse()``` to allow users to define specific hyperparameters in the script.
  - This might include e.g. learning rate, batch size, etc

### Personal Learning Goals
Apart from continuing my exploration of Github Copilot and the SOLID principles, I want to see how much I can transfer from the [assignment 2](../vision-a2/) to this one wrt the structure. I will still create individual files to reduce coupling, but it might be fun to see. 

## Methods and Design
The overall approach to this task is quite similar to [assignment 2](../vision-a2/) with a few different notes. the main difference is that we use a much more powerful system in relying on vgg16 as our base model.

Other than that the flow goes as follows (as shown in [transfer_cnn.py](./transfer_cnn.py)): 
1. specify the hyperparameters using the command line
2. load the data 
3. initialize the model
4. train the model
5. evaluate (both in terms of the training plot and the classification report)

### Software Design
As mentioned, I have tried to follow the SOLID-principles. On a high-level, this means that the main script (`transfer_cnn.py`) is easy to modify to suit future needs as it is relatively agnostic to implementation details. This means that it would be relatively easy to change the base model to another model than VGG16, though it would take a bit of refactoring to make this process silky smooth. Below are some concrete examples of the SOLID principles.
- **Single responsibility**: Each function attempts to do just one thing, which decreases coupling.
- **Interface segregation**: Each functionality (such as loading data or reporting) has a separate file, which decreases the dependencies between scripts. This also means that the main script only imports other files and base modules.
- **Liskov substitution**: Not strictly implemented as there are no classes. However, the design utilizes that tensorflow uses Liskov substitution, which makes extensions easier.
- **Open-closed**: Functions are build in a modular way that makes them relatively easy to extend with e.g. new models.
- **Dependency Inversion**: Not used too much, as the task only requires to work with VGG16. However, for possible extensions this could make adding other models quite a bit easier.

## Usage 
TL;DR: An example of the entire setup and running the pipeline can be run using the bash-script `run_project.sh`. 

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script
The script has two arguments `--batch-size` for controlling the batch size (useful for configuring to different memory sizes) and `learning-rate` for controlling the learning rate. The defaults are sensible but can be changed as below.

```console
$ python transfer_cnn.py --batch-size 32 --learning-rate 0.01
```
## Discussion and Results
### Results
#### Loss and Accuracy 
**Loss**

[loss plot](./output/plot_loss.png)

**Accuracy**

[accuracy plot](./output/plot_accuracy.png)

#### Classification Report

### Discussion
- still bad performance
- can see the training loss levelling off --> symptom of underfitting 
- Bigger model
    - Some unsuccessful experiments with adding fully connected layers --> overfit to one class
- not enough data?
    - image augmentation
    -  