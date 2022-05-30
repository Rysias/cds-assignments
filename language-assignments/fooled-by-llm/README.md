# Self Assignment - News Generation with GPT
[GITHUB LINK](https://github.com/Rysias/cds-assignments/tree/main/language-assignments/fooled-by-llm)

![](https://cdn.iflscience.com/images/4421c427-d24f-5e92-9176-5ce473fe5626/default-1567161944-cover-image.jpg)
*Source: cdn.iflscience.com*

Throughout these assignments we have looked at increasingly sophisticated NLP techniques from [simple string processing](../language-a1/README.md) to [deep learning](../language-a4/README.md). In this self-assignment we will look at the current hottest topic in NLP, namely, large generative models. 

Large generative models are all the rage in NLP. Starting with the (now infamous) 1.5 billion parameter [GPT-2 from OpenAI](https://openai.com/blog/gpt-2-1-5b-release/), the AI community discovered that merely throwing larger models, more data, and more compute at so-called foundation models increased performance on a wide range of tasks from [machine translation](https://paperswithcode.com/sota/machine-translation-on-wmt2014-english-german) to [question answering](https://paperswithcode.com/dataset/squad). And this was with only minimal fine-tuning!

While these models can be used for good, they can also be employed for more nefarious purposes. As their main objective is [mimicking human text](https://dl.acm.org/doi/10.1145/3442188.3445922), they can be used to automatically generate content - without any moral compas to guide. In this post-truth era it is not difficult to see how that might go accelerate already worrying developments. 

This assignment investigates the capabilities of these language models to generate news. Ultimately, the goal is to create a kind of news [*Turing test*](https://www.techtarget.com/searchenterpriseai/definition/Turing-test) to evaluate how well these generative models can generate fake and "true" news respectively. This assignment will serve as the initial step towards that goal.  



## Table of Content
- [Assignment Description](#assignment-description)
    * [Personal Learning Goals](#personal-learning-goals)
- [Methods](#methods-and-design)
    * [Data Collection](#data-collection)
    * [Cleaning News](#cleaning-news)
    * [Models](#models)
    * [Prompt Engineering](#prompt-engineering)
- [Software Design](#software-design)
- [Usage](#usage)
    * [Getting an API Key](#getting-an-api-key)
    * [Setting Up](#setting-up)
    * [Using the Script(s)](#using-the-scripts)
- [Discussion](#discussion)

## Assignment Description
For this assignment the goal is to write software for generating the beginning of news articles based on headlines using GPT-models. The software should do the following: 

1. Download (and clean) a suitable dataset
    - This script can then be used as a ground-truth dataset for the turing
2. Create a script for generating news articles  
    - The script should take as input a) the source dataset and b) the model to use
3. Format the data to be ready for a News Turing test

### Personal Learning Goals
There are two learning goals in particular that have shaped my development of this assignment. One is to gain experience working with large language models, which I (and other [smarter people](https://openai.com/blog/better-language-models/)) believe might be the future of Artificial (General) Intelligence. The second is to create a workflow and structure that supports rapid iterations and reproducibility. This is particularly important when exploring new technologies but iteration speed and reproducibility are key skills across all of data science (and science more broadly). 

## Methods and Design
### Data Collection
The dataset I use is the [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/assets/docs/ISOT_Fake_News_Dataset_ReadMe.pdf). It is a large collection of "real" news scraped from Reuters.com and fake news scraped from a wide variety of internet sources. Furthermore, the data is easily accessible via kaggle.com and the script [`download_data.sh`](/data_download.sh)

### Cleaning News
While the ISOT dataset is in general relatively clean, it is nevertheless an NLP dataset which necessitates more cleaning. The cleaning pipeline consists of both general and domain specific steps. 

The general cleaning steps are quite standard and minimal: basically it consists of normalising whitespace. As the end-consumer are human readers most of the standard preprocessing pipeline such as removing punctuation, lower-casing and lemmatisation have been left out. 

On the domain-specific side, there are a few steps that needed to be taken to ensure good prompt generation and fair comparison: 
- **Removing 'tags'**: The headlines and texts would have different markers that would bias the prompt generation and make them easier to distinguish. This could be words like "VIDEO" in the title, or the fact that all Reuters articles start with the location of the story (e.g. "WASHINGTON (REUTUERS)...")
- **Removing URLs**: While GPT models can generate URLs these are most often invalid. Therefore, I have removed these from the texts as a) it makes the comparisons more fair and b) actual fake news generation would probably do the same. 
- **Limiting length of articles**: I have limited the text of the ground-truth and generated articles to a maximum of two sentences (or <75 tokens). I will discuss the reasons for this in other sections. 


### Models
This project uses open source models trained by [EleutherAI](https://www.eleuther.ai/faq/), an open collective dedicated to training and understanding large language models. They have trained a bunch of GPT models ranging from the relatively small (125 million parameters) to the fairly huge (20 billion parameters). 

As all of these are require too much compute and memory for personal computers (and even UCloud), the project relies on API calls from [goose.ai](goose.ai). Goose.ai is an extremely cheap and easy to use provider of EleutherAI's models. They integrate well with the OpenAI python library making it relatively easy to implement. 

Relying on the API also makes the project extremely easy to scale. The difference between using a small (and cheap) model such as `gpt-neo-125m` and a huge model 20 billion parameter model like [GPT NeoX 20B](https://blog.eleuther.ai/announcing-20b/) is literally just one parameters. 

### Prompt Engineering
Figuring out good prompts is an [essential component of creating good output](https://blog.andrewcantino.com/blog/2021/04/21/prompt-engineering-tips-and-tricks/) from large language models. For experimenting I have used [6b.eleuther.ai](https://6b.eleuther.ai/), a GUI created by the EleutherAI teams which allows for rapid testing of different ideas - without having to pay for use!

The prompt that I ended up using was the following: 
```
{Real|Fake} headline: {headline from dataset}
text: 
```
This has the aim of giving the model some context wrt the style (real or fake news) as well as the type of text to generate (text following a headline). 

For the `temperature` parameter, which controls how 'risky' the model is when choosing answers with 0 being argmax-sampling and 1 being the maximum, I choose 0.9 which is the [recommended value by goose.ai](https://goose.ai/docs). 


### Software Design
The main goal of the software design is to balance flexibility and rapid iteration with reproducibility. There are several choices that supports this. 

One is to use scripts instead of notebooks. While, I have used notebooks for EDA ([`ExploreDataset.ipynb`](./ExploreDataset.ipynb)) and figuring out how to clean the data ([`CleanPrompts.ipynb`](./CleanPrompts.ipynb)), the entire pipeline can be run as scripts as crystallized in [`run_project.sh`](./run-project.sh). 

Secondly, I have used [argparse](https://docs.python.org/3/library/argparse.html) to provide flexible documentation and usage for the different scripts. This allows users to experiment while still having strong defaults. 

Thirdly, I have used [test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development) for part of the workflow. This ensures that the code is stable and design is put at the forefront.

Finally (and perhaps most importantly), I have designed by following the SOLID-principles. As in the other assignments, these guiding principles make for cleaner code overall. Below is a breakdown. 

- **Single responsibility**: Each aspect of the code (like [cleaning files](./src/clean_text.py) or [generating prompts](./src/prompts.py)) are split out into seperate components.
- **Open-closed**: By using argparse, it becomes easier to change the behavior without modifiying the code. Furthermore, relying on the API makes it super easy to change back-ends. 
- **Liskov substitution**: Not applicable as I don't use object oriented design.
- **Interface segregation**: By splitting the functionality into a `src/` directory with separate functionality the main scripts are relatively indifferent to implementation details. 
- **Dependency Inversion**: The OpenAI API makes it possible to pass different models instead of writing custom applications which makes for a cleaner design. Furthermore, I use [dependency injection](https://stackify.com/dependency-injection/) to make it easier to pass different prompt-generation functions in [`generate_news.py`](/generate_news.py)

## Usage 
TL;DR: An example of the entire setup and running the pipeline can be run using the bash-script `run_project.sh`. however, you need to have a valid API in a `config.json` for this to work. 

### Getting an API Key
To use the main script, `generate_prompts.py`, you need an API key from goose.ai. There are a couple of steps to get one: 
1. setup your account on [goose.ai](https://goose.ai/docs) by following the linked guide
2. Copy your secret key to a file called `config.json`. The json should look like the one below:

    ```{json}
    {
        "goose_api": <YOUR API KEY HERE>
    }
    ```

### Setting Up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the Script(s)
There are two main scripts for this project: `generate_news.py` and `prompt_pipeline.py`. 

[`generate_news.py`](/generate_news.py) is the main script for generating news based on headlines and other information using the generative models. It is documented via argparse, which makes it possible to use `python generate_news.py --help` to get up to date documentation. The output looks like this: 
```console
usage: generate_news.py [-h] [--model-name {gpt-neo-125m,gpt-j-6b}] [--prompt-function {type_title_prompt,type_title_date_prompt}] [--file-path FILE_PATH]
                        [--max-tokens MAX_TOKENS] [--temperature TEMPERATURE]

Generate news using a GPT model from goose.ai

optional arguments:
  -h, --help            show this help message and exit
  --model-name {gpt-neo-125m,gpt-j-6b}, -m {gpt-neo-125m,gpt-j-6b}
                        GPT model to use for generation (default: gpt-neo-125m)
  --prompt-function {type_title_prompt,type_title_date_prompt}, -p {type_title_prompt,type_title_date_prompt}
                        Function to use for generating prompts (default: type_title_prompt)
  --file-path FILE_PATH, -f FILE_PATH
                        Path to the source for generating prompts (default: data/clean_news_examples.csv)
  --max-tokens MAX_TOKENS, -t MAX_TOKENS
                        Maximum number of tokens to generate (default: 75)
  --temperature TEMPERATURE, -T TEMPERATURE
                        Temperature for GPT model (default: 0.9)
```

Most of the arguments have strong defaults but can be easily changed for quick iteration. 

The script [`prompt_pipeline.sh`](/prompt_pipeline.sh) runs the entire pipeline for creating a generative news Turing test. This runs the following scripts: 
1. `create_news_examples.py`: For randomly selecting entries from the raw file and creating the raw output. 
2. `clean_news.py`: For cleaning the raw data as described in [the methods section](#cleaning-news). 
3. `generate_prompts.py`: As described above
4. `clean_prompts.py`: For cleaning the prompts similar to the news pipeline. 
5. `split_dataset.py`: For splitting the dataset to allow proper statistical matching. 

It takes only one argument which is the model name. This allows the developer to test the pipeline on a cheap model and run it on a more expensive (and better) one.

#### **Example Usage**
```console
$ python generate_news.py --model-name "gpt-j-6b" --temperate 0.95 --max-tokens 1000
```
## Discussion
The best way to illustrate the abilities of the generative model (and this project in general) is to investigate a couple of examples. All examples are generated by `gpt-j-6b` with a simple type + headline prompt.

Let's start with a (true) headline from Reuters:
```
U.S. says it will take steps after Cambodia's dissolves opposition party
```
The ground-truth goes as follows: 
```
"The United States expressed  grave concern  on Thursday about the Cambodian government s decision to dissolve the main opposition party and said Washington will take  concrete steps  in response, according to a White House statement. As a first step, the White House said, the United States will end its support for the Cambodian National Election Committee and its administration of the 2018 national election."
```
And the GPT generated text: 
```
The U.S. said that it was preparing a list of those responsible for the 'transgressions' and possible sanctions as a warning to the Cambodian government to respect democratic freedoms, including the dissolution.
```

At first glance the GPT-text looks quite convincing; there are no blatant grammatical errors and the text sounds plausible. However, there is a huge problem: the text is not grounded in facts. It is not certain that the U.S. were actually preparing a list with responsible people. In fact, they were instead "ending support for the [...] Election Committee". 

This ability is what has made large language models into "Stochastic Parrots" ([Bender et al., 2021](https://dl.acm.org/doi/10.1145/3442188.3445922)). They learn to mimic what they have seen (human generated data), without having their knowledge grounded in reality. This is illustrated by the fact that larger language models perform _worse_ on the benchmark [TruthfulQA](https://owainevans.github.io/pdfs/truthfulQA_lin_evans.pdf)

However, from this experiment I also found that the models produce a lot of bad text. Fictive URLs, weirdly formatted emails, pure repetitions, and other useless output. I often find myself cherry-picking good examples, or perhaps lightly editing the texts. 

This points to a more fundamental discussion about the interaction of humans and machines in research and applications. For what amount of manual editing is "fair"? Is it still machine generated text if a human editor has chosen the one coherent example out of thousands? 

My personal take is that minor copy editing is acceptable when doing research. As most people who have tried writing knows, a blank page is much more than an article to review. Machine generated text might therefore be more valuable (and more harmful) when used as a rough generator than if used an end-to-end pipeline. 

That being said, I only scratched the surface of prompt engineering in this assignment. With more dedicated work (and the [accelerating pace of model improvements](https://hbr.org/2021/09/ai-adoption-skyrocketed-over-the-last-18-months)), who knows how good the text generation can get? One thing is certain: it is essential for the humanities and social sciences to be in the room to evaluate the progress. 
