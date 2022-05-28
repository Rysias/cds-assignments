# Assignment 3 - Transfer learning + CNN classification
[GITHUB LINK](https://github.com/Rysias/cds-assignments/tree/main/language-assignments/fooled-by-llm)

The project uses GPT (Generative Pre-trained Transformer) models. The main idea behind these is to train a transformer (LINK) on a huge pile of text. This has (perhaps surprisingly) shown to be an extremely effective approach in making models perform (relatively) well on a wide variety of tasks from [machine translation](LINK) to [question answering](LINK) - sometimes even [without extra training](LINK)! 

There are, however, several complications with running these models. Because of their extreme size, they require extraordinary resources to run. GPT-3, the famous model trained by OpenAI, has 175 BILLION parameters requiring a whooping [700GB of memory](https://www.hyro.ai/glossary/gpt-3))(!). This requires multiple GPUs (or [TPUs](https://www.hyro.ai/glossary/gpt-3)) just for inference, which is more than most people (or even organisations) can handle. 
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
For this assignment the goal is to write software for generating the beginning of news articles based on headlines using GPT-models. The software should do the following: 

1. Download (and clean) a suitable dataset
    - This script can then be used as a ground-truth dataset for the turing
2. Create a script for generating news articles  
    - The script should take as input a) the source dataset and b) the model to use
3. Format the data to be ready for a News Turing test

### Personal Learning Goals
There are two learning goals in particular that have shaped my development of this assignment. One is to gain experience working with large language models, which I (and other smarter people (LINK)) believe might be the future of Artificial (General) Intelligence. The second is to create a workflow and structure that supports rapid iterations and reproducibility. This is particularly important when exploring new technologies but iteration speed and reproducibility are key skills across all of data science (and science!). 

## Methods and Design
### Data Collection
# TODO: IMPLEMENT KAGGLE SCRIPT
The dataset I use is the [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/assets/docs/ISOT_Fake_News_Dataset_ReadMe.pdf). It is a large collection of "real" news scraped from Reuters.com and fake news scraped from a wide variety of internet sources. Furthermore, the data is easily accessible via kaggle.com and the script [`download_data.sh`]()

### Cleaning News
While the ISOT dataset is in general relatively clean, it is nevertheless an NLP dataset which necessitates more cleaning. The cleaning pipeline consists of both general and domain specific steps. 

The general cleaning steps are quite standard and minimal: basically it consists of normalising whitespace. As the end-consumer are human readers most of the standard preprocessing pipeline such as removing punctuation, lower-casing and lemmatisation have been left out. 

On the domain-specific side, there are a few steps that needed to be taken to ensure good prompt generation and fair comparison: 
- **Removing 'tags'**: The headlines and texts would have different markers that would bias the prompt generation and make them easier to distinguish. This could be words like "VIDEO" in the title, or the fact that all Reuters articles start with the location of the story (e.g. "WASHINGTON (REUTUERS)...")
- **Removing URLs**: While GPT models can generate URLs these are most often invalid. Therefore, I have removed these from the texts as a) it makes the comparisons more fair and b) actual fake news generation would probably do the same. 
- **Limiting length of articles**: I have limited the text of the ground-truth and generated articles to a maximum of two sentences (or <75 tokens). I will discuss the reasons for this in other sections. 


### Models
This project uses open source models trained by [EleutherAI](LINK), an open collective dedicated to training and understanding large language models. They have trained a bunch of GPT models ranging from the relatively small (125 million parameters) to the fairly huge (20 billion parameters). 

As all of these are require too much compute and memory for personal computers (and even UCloud), the project relies on API calls from [goose.ai](goose.ai). Goose.ai is an extremely cheap and easy to use provider of EleutherAI's models. They integrate well with the OpenAI python library making it relatively easy to implement. 

Relying on the API also makes the project extremely easy to scale. The difference between using a small (and cheap) model such as `gpt-neo-125m` and a huge model 20 billion parameter model like [GPT NeoX 20B](LINK) is literally just one parameters. 

### Prompt engineering
Figuring out good prompts is an essential component of creating good output from large language models (LINK). For experimenting I have used [6b.eleuther.ai](https://6b.eleuther.ai/), a GUI created by the EleutherAI teams which allows for rapid testing of different ideas - without having to pay for use!

The prompt that I ended up using was the following: 
```
{Real|Fake} headline: {headline from dataset}
text: 
```
This has the aim of giving the model some context wrt the style (real or fake news) as well as the type of text to generate (text following a headline). 

For the `temperature` parameter, which controls how 'risky' the model is when choosing answers with 0 being argmax-sampling and 1 being the maximum, I choose 0.9 which is the recommended value by goose.ai (LINKS). 


### Software Design
The main goal of the software design is to balance flexibility and rapid iteration with reproducibility. There are several choices that supports this. 

One is to use scripts instead of notebooks. While, I have used notebooks for EDA ([`ExploreDataset.ipynb`](./ExploreDataset.ipynb)) and figuring out how to clean the data ([`CleanPrompts.ipynb`](./CleanPrompts.ipynb)), the entire pipeline can be run as scripts as crystallized in [`run_project.sh`](./run-project.sh). 

Secondly, I have used [argparse](LINK) to provide flexible documentation and usage for the different scripts. This allows users to experiment while still having strong defaults. 

Thirdly, I have used test-driven development (TDD; LINK) for part of the workflow. This ensures that the code is stable and design is put at the forefront.

Finally (and perhaps most importantly), I have designed by following the SOLID-principles. As in the other assignments, these guiding principles make for cleaner code overall. Below is a breakdown. 

- **Single responsibility**: Each aspect of the code (like [cleaning files](./src/clean_text.py) or [generating prompts](./src/prompts.py)) are split out into seperate components.
- **Open-closed**: By using argparse, it becomes easier to change the behavior without modifiying the code. Furthermore, relying on the API makes it super easy to change back-ends. 
- **Liskov substitution**: Not applicable. 
- **Interface segregation**: By splitting the functionality into a `src/` directory with separate functionality the main scripts are relatively indifferent to implementation details. 
- **Dependency Inversion**: The OpenAI API makes it possible to pass different models instead of writing custom applications which makes for a cleaner design.

## Usage 
TL;DR: An example of the entire setup and running the pipeline can be run using the bash-script `run_project.sh`. however, you need to have a valid API in a `config.json` for this to work. 

### Getting an API key
To use the main script, `generate_prompts.py`, you need an API key from goose.ai. There are a couple of steps to get one: 
1. setup your account on [goose.ai](https://goose.ai/docs) by following the linked guide
2. Copy your secret key to a file called `config.json`. The json should look like the one below:

    ```{json}
    {
        "goose_api": <YOUR API KEY HERE>
    }
    ```

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script(s)
# TODO 
- Bash script 
- Prompt script 
# TODO 

Parameter | Type | Required | Description
---- | ---- | ---- | ----


#### Example usage
```console
$ python 
```
## Discussion

Take the example of the Reuters headline: 
```
"Obama, Mexican president discuss immigration, anti-drugs fight: White House"
```
The ground-truth goes as follows: 
```
"U.S. President Barack Obama and Mexican President Enrique Peña Nieto discussed immigration from Central America and the fight against heroin production during a phone call on Thursday, the White House said. “ The leaders committed to continue working jointly to address irregular migration from Central America,” the White House said in a statement. “"
```
And the GPT generated text: 
```
"President Obama met with Mexican President Felipe Calderon as part of their exchange of prime ministers. Obama’s first official trip abroad as president included a stop at a camp for children and young immigrants. He urged them to learn about the country’s political system and find a job." (...)
```

....

Contrast that with the Fake news headline:
```
'“This is a big deal!” Obama Lied About ISIS Progress In The Middle East'
```
Here the ground-truth text goes:
```
"Steve Hayes says the biggest scandal yet for Obama is possibly the downplaying of the progress of ISIS in the middle east. Basically, ISIS intel was cooked to make Obama look good crazy!"
```
And the generated text is: 
```
"“ISIS is effective, well-funded and expanding, taking over the U.S.-backed regional allies like our own Iraqi Embassy in Amman, Jordan. The White House is lying.” (...)
```


- Examples (in a table; semi manual)
    - Comments on the examples
- Notes on the process
    - Fake URLs 
    - Tendency to repeat itself
- The role of manual selectoin

# TODO List
- [ ] Fill out TODO's in this readme 
- [ ] Add download data (kaggle) --> data/raw
- [ ] Find way to visualize.