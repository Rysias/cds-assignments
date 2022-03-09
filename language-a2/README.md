# Analyzing Fake and Real News
This repository is for extracting Geopolitical Entities (GEOP) and analyzing sentiment from headlines of real- and fake news. The project is structured as a mini "package" with functions in `src/news_entities.py`, tests in `tests/`, and the main script in the main folder (`process_news.py`). 

## Usage 
The script allows one to a) specify the path to the data, b) the amount of entities to plot, and c) which sentiment  backend to use ([vader](https://github.com/cjhutto/vaderSentiment) or [textblob](https://github.com/sloria/TextBlob)). For usage guide, execute `python process_news.py --help` in the terminal. And example of getting the top 30 entities and using the vader-backend looks like this 

```console
$ python process_news.py --sentiment vader --top-n 30
```

## Other files
Other files not previously described include
- `experiments.ipynb`: for interactively testing the functionality and designing (not up to date)

## Testing
The scripts were developed using a TDD-methodology using [pytest](https://docs.pytest.org/en/7.0.x/). To execute the test suite run `python -m pytest` from the main directory (`language-a2`)


# TODO
- [x] Create full run function
- [x] Save sentiment function in file names
- [x] Documentation
- [x] Finish README
- [x] Move functions to separate script