"""
- Take a user-defined search term and a user-defined window size.
- Take one specific text which the user can define.
- Find all the context words which appear ± the window size from the search term in that text.
- Calculate the mutual information score for each context word.
- Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.
"""
from spacy import load
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

NLP = load("en_core_web_sm")
DATA_DIR = Path("../../../CDS-LANG/100_english_novels/corpus")
assert DATA_DIR.exists()


def read_txt(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()
    
def get_window(words, idx, window_size):
    return words[idx-window_size:idx] + words[idx+1:idx+window_size+1]

def flatten_list(lst):
    return [x for sub in lst for x in sub]

def count_words(corpus, term): 
    return sum(1 for word in corpus if word == term)

def clean_file(file_path):
    raw_text = read_txt(file_path)
    return " ".join(raw_text).lower()

def tokenize_doc(text):
    tokenizer = NLP.tokenizer
    return tokenizer(text)
    
def get_doc(file_path):
    clean_text = clean_file(file_path)
    return tokenize_doc(clean_text)

def get_word_list(doc): 
    return [tok.text for tok in doc if not tok.is_punct | tok.is_space]

def find_target_idx(words, search_term):
    return [i for i, word in enumerate(words) if word==search_term]

def find_collocates(corpus, search_term_idx, window_size):
    all_collocates = [get_window(corpus, idx, window_size) for idx in search_term_idx]
    return flatten_list(all_collocates)

def create_collocate_df(corpus, search_term, window_size):
    search_term_idx = find_target_idx(corpus, search_term)
    all_collocates = find_collocates(corpus, search_term_idx, window_size)
    collocate_counts = Counter(all_collocates)
    return (pd.DataFrame.from_dict(collocate_counts, orient='index')
                        .reset_index()
                        .rename(columns ={"index":"collocate", 0:"collocate_count"}))

def calc_mi(collocate_count, corpus_freq, corpus, search_term, window_size): 
    corpus_size = len(corpus)
    node_freq = count_words(corpus, search_term) 
    return np.log10((collocate_count * corpus_size) / (node_freq * corpus_freq * window_size * 2)) / np.log10(2)

def collocate_pipeline(corpus, search_term, window_size):
    df = create_collocate_df(corpus, search_term, window_size)
    df['corpus_count'] = df['collocate'].apply(lambda x: count_words(corpus, x))
    df["MI"] = calc_mi(df['collocate_count'], df['corpus_count'], corpus, search_term, window_size) 
    return df

def process_file(file_path, search_term, window_size): 
    doc = get_doc(file_path)
    corpus = get_word_list(doc)
    return collocate_pipeline(corpus, search_term, window_size)
    
def write_output(collocate_df, file_name, search_term): 
    output_name = Path('output') / f"{file_name[:-4]}_{search_term}.csv"
    collocate_df.to_csv(output_name, index=False)
    
    
def main(args):
    search_term = args.search_term
    window_size = args.window_size
    file_path = DATA_DIR / args.file_name
    
    collocate_df = process_file(file_path, search_term, window_size) 
    write_output(collocate_df, args.file_name, search_term)
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description = "Creates collocation info for a given file, search term, and window size. The output is written to a csv with the name '{text_name}_{search_string}.csv' in the output folder")
    argparser.add_argument("--file-name", required=True, help="file-name to search through (in 100_english_novels")
    argparser.add_argument("--search-term", required=True, help="Node word to find collocates")
    argparser.add_argument("--window-size", required=True, type=int, help="Window size (on each side of node word)")
    args = argparser.parse_args()
    main(args)