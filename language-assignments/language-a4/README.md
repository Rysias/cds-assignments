# Toxic Classification
This assignment compares two different classifiers, a TF-IDF based logistic regression and a fine-tuned convolutional neural network, on the THREAT dataset from Hammer et al. (2019).

The dataset is highly imbalanced. Therefore, I use [imbalanced-learn](https://imbalanced-learn.org/) to alleviate the problems with random under-sampling. 

The logistic regression uses a highly opinionated TF-IDF that removes stopwords and has a feature cap of 1000. The dataset is also cleaned with lowercasing, punctuation removal, and whitespace. Otherwise, no substantial feature engineering is currently implemented. 

The neural network is based on transfer-learning from the [nnlm-en-dim50-with-normalization](https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2) model. This has two advantages: 1) it takes care of the pre-processing using the knowledge of smart google engineers, and 2) you get a really powerful transfer-learning model based on the [English Google News 7B corpus.](LINK NEEDED). (2) is especially nice since the THREAT dataset shares some of the same topics. For the implementation I have followed [this guide](https://www.tensorflow.org/hub/tutorials/tf2_text_classification). I have, however, added random under-sampling to avoid underfitting and changed the architecture to reflect the modest size of the dataset. 

Though I have drawn on inspiration from the above guide, I made the workflow more SOLID as can be seen in the software design section.

## Software Design
The software design attempts to follow the SOLID principles. The concrete 

- **Single responsibility**: I have split the functionality into files to make sure that there is high cohesion and low coupling.
- **Interface segregation**: By using type hints, most functions are clear about what they expect. Also, the main functions are kept fairly clean with minimal imports, so everything is pretty neat.
- **Liskov substitution**: Not quite applicable as the design is more functional than object oriented.
- **Open-closed**: The code is designed in such a way that the dataset is relatively arbitrary - as long as there is a text-column and a target column called "label", the stuff will work. 
- **Dependency Inversion**: Not explicitly used but most functions are easy to replace (i.e. having another vectorizer)

In general, I have prioritized having highly opinionated stuff rather than giving much flexibility. However, that would be relatively easy to change given the design.

## Usage 
TL;DR: An example of the entire setup, testing, running pipeline can be run using the bash-script `run_project.sh`. 
### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script
The easiest way to figure out how the scripts work is using the `--help` flag with each of them. Nevertheless, below is an example of the neural network one
```console
$ python dnn_text.py --dataset "../donald_trump_tweets.csv" --epochs 100
```

## Other files
Other files not previously described include:
- `experiments.ipynb`: for interactively testing the functionality and designing (not up to date)


## References
Hammer, H. L., Riegler, M. A., Øvrelid, L. & Veldal, E. (2019). "THREAT: A Large Annotated Corpus for Detection of Violent Threats". 7th IEEE International Workshop on Content-Based Multimedia Indexing.