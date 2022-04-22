# Simple Image Classifications 
TODO: Simple description


## Software Design (TODO: This)
- **Single responsibility**: 
- **Interface segregation**: 
- **Liskov substitution**: 
- **Open-closed**: 
- **Dependency Inversion**:

## Usage 
TL;DR: An example of the entire setup and running the pipeline can be run using the bash-script `run_project.sh`. 

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script
The script has two arguments `--batch-size` for controlling the batch size (useful for configuring to different memory sizes) and `learning-rate` for controlling the learning rate. The defaults are sensible but can be changed as below.

```console
$ python transfer_cnn.py --batch-size 32 --learning-rate 0.01
```

## References and Resources
https://www.tensorflow.org/tutorials/images/transfer_learning
https://www.tensorflow.org/tutorials/images/cnn
https://medium.com/codex/github-copilot-is-a-game-changer-cd0a2bbe6de8
