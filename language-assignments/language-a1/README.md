# Assignment 1: Collocation of words in corpus
This folder contains the scripts for finding collocation in a txt-file given a search-term and a window. All solutions are python scripts with cmd-arguments. Documentation for each script can be found using the `--help` flag. In addition to the scripts, there is a jupyter notebook for experimenting with the different solutions. The tokenization scheme for all is lower case and disregarding punctuation (filtered through SpaCy). SpaCy is the main tool used together with pandas/numpy. 

## Scripts
1. `collocate_single_text.py`: solves the basic task described above. 
2. `collocate_all_texts.py`: solves the extra task of generating one csv per text in the entire corpus
3. `collocate_corpus.py`: solves the extra task of analyzing the whole corpus as one text
4. `collocate_multi_words.py`: solves the extra task of multiple search terms (in one text)

## Other files
- `collocate_experiments.ipynb`: code sketches and experiments
- `profile_collocate.sh`: Script for measuring performance of collocates (used for finding optimizations)
- `*.txt`: outputs from `profile_collocate_sh`

# TODO
- [x] refactor functions into src
- [x] create reproducible pipenv
- [x] create run_project.sh 
- [ ] Create README following structure of the other repos