# Assignment 3 - Transfer learning + CNN classification
[GITHUB LINK](https://github.com/Rysias/cds-assignments/tree/main/language-assignments/language-a4)

# TODO 

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
The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder ```CDS-LANG/toxic``` and trying to see if we can predict whether or not a comment is a certain kind of *toxic speech*. You should write two scripts which do the following:

- The first script should perform benchmark classification using standard machine learning approaches
  - This means ```CountVectorizer()``` or ```TfidfVectorizer()```, ```LogisticRegression``` classifier
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of deep learning methods we saw in class
  - Keras ```Embedding``` layer, Convolutional Neural Network
  - Save the classification report to a text file 

### Personal Learning Goals
# TODO 

## Methods and Design
### Data
What make this task difficult is the distribution of the labels. As can be seen from table 1 below, there are very few positive examples (approximately 5%). 

|Threatening Comments | Normal Comments |
| :---:  | :---: |  
| 1,387 | 27,256 |
*Table 1: Distribution of labels*

This imbalance have implications for both the evaluation metrics and the training. With regards to evaluation metrics, accuracy becomes a poor candidate. Imagine a naive classifier that classifies all comments as non-threatening. This would have a 95% accuracy which is obviously not reflective of the value delivered. 

Instead, breaking down the metric into precision and recall gives a more nuanced view of false positives and -negatives. I will therefore focus on these when we reach the [discussion](#discussion). 

The implications for the training data is that one should make sure to have the training procedure reflect the above to avoid a biased estimator. I will discuss this further for the two models below. 

### Baseline: logistic regression
For the baseline, I have chosen a simple setup. I have implemented a "pipeline" with the following components

1. [Undesampler](LINK): Randomly undersamples the majority class to get equally many positive and negative examples
2. [TF-IDF](link): Creates TF-IDF features (link), disregarding stop words
3. [Logistic Regression](link) 

I create these using scikit-learn and imbalanced-learn pipelines, which abstracts away many of the difficulties creating valid classifiers such as [data leakage](LINK). 

### Deep Learning approach
The main limitation of the baseline appraoch is that TF-IDF only uses individual words while disregarding context such as semantics and syntax. More modern deep learning approaches take care of this by creating end-to-end models trained on huge amounts of data. Empirically, this creates more sophisticated features which in turn create better models. (LINKS). 

For this assignment I use transfer learning with the `nlm-en-dim50-with-normalization` from [google](https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2) as my feature layer. This uses a feed-forward neural network trained on the Google 7B news dataset to create a 50-dimensional embedding, which we can then build a classifier on top of. 

There are several advantages of this approach (in theory): 
- It handles the preprocessing of the text data
- It (hopefully) creates sophisticated features which we can use for classification
- It is (relatively) simple to implement. 

However, for this to work well on our use case we need to take the imbalanced data seriously to avoid the default prediction scenario. There are several steps I will attempt: 

1. **Undersampling**: just like in the [logistic regression](#logistic-regression) described above. 
2. **Fake data generation**: I the package [nlpaug](LINK) to generate extra training data by replacing synonyms in the toxic dataset. 
3. **Adding training data from the net**: I add extra toxic training data from the [jigsaw kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) to reduce the imbalance.

Whether these approaches will work, we will discover in the [discussion section](#discussion).

### Software Design
# TODO 
- **Single responsibility**: 
- **Open-closed**: 
- **Liskov substitution**: 
- **Interface segregation**: 
- **Dependency Inversion**: 

## Usage 
TL;DR: An example of the entire setup and running the pipeline can be run using the bash-script `run_project.sh`. 

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script
# TODO 

Parameter | Type | Required | Description
---- | ---- | ---- | ----


#### Example usage
```console
$ python 
```
## Discussion and Results
### Classification Reports
#### Logistic Regression
```{console}
              precision    recall  f1-score   support

           0       0.99      0.91      0.95      5441
           1       0.31      0.81      0.45       279

    accuracy                           0.90      5720
   macro avg       0.65      0.86      0.70      5720
weighted avg       0.96      0.90      0.92      5720
```

#### Neural Network
```{console}
              precision    recall  f1-score   support

           0       0.95      1.00      0.98      5453
           1       1.00      0.00      0.00       276

    accuracy                           0.95      5729
   macro avg       0.98      0.50      0.49      5729
weighted avg       0.95      0.95      0.93      5729
```
### Discussion
To paraphrase the opening line of Anna Karenina: "All well-behaving models are alike, but every misbehaving model misbehaves in its own way." This dataset proved to be an incredibly tough challenge, especially for the deep learning approach. Therefore, this discussion will primarily serve as a post-mortem of my (many) attempts to make this work. 

Let's start by investigating the logistic baseline in the table above. The most interesting metric for this particular use case is the F1-score, and particularly the F1 score for the positive class and the macro average f1 score. The reason we care about F1 (rather than accuracy) is that we want to balance false positives and false negatives. We neither want to remove as many potential threats as possible without stiffling the conversation. 

We can see that the logistic regression is a bit trigger happy - the positive recall is pretty good (0.81) while the positive precision is fairly poor (0.31). This results in a not great positive F1 score of 0.41 and a macro F1 score of 0.7. 

Compared to the neural network, however, these values are wonderful! From the table it becomes clear that the model has learned the rule that there are no threats in the dataset whatsoever, which results in classifying every comment as non-threatening. This gives the model great accuracy but a dismal positive F1 score of 0.0. 

The reason is - of course - the imbalanced dataset. While there are many techniques to alleviate the problem none of them worked. I've tried undersampling, oversampling with synthetic data, changing the architecture, adding regularization, removing regularization, adding additional data, and a ton of small parameter tweaks. All in vain. 

If assignments are truly about the journey, not the destination, this assignment has been great. However, if the goal is to implement an actual threat detection model here is how I would do it: 

1. Install the [detoxify](https://github.com/unitaryai/detoxify)-package
2. Run predictions on VideoCommentsThreatCorpus to extract the threat scores.
3. Tune the threshold to strike the right balance between false positives and false negatives. 

The above approach highlights some interesting points about the direction of data science. The approach is, from an engineering perspective, extremely easy to implement: It requires little to no pre-processing, and the model is implemented in 2 lines of code (LINK). However, behind that is a huge structure of shoulders of giants: the company, Unitary, who has developed the Detoxify package, which relies heavily on the transformers library by huggingface, which in turn relies on the research in large language models driven by huge tech giants like Google, Microsoft, and Facebook. 

Where does this leave the digital humanities scholar / data scientist? In many ways it leverages their value: as ever more technical detail is abstracted away, how the models are used becomes increasingly important. This requires interdisciplinary skills, where one understands both the technical implementations and the societal implications. I will investigate more of this in [my self-assignment](../fooled-by-llm/), where I will look at truly large language models. 





