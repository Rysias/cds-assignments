"""
- Take a user-defined search term and a user-defined window size.
- Take one specific text which the user can define.
- Find all the context words which appear ± the window size from the search term in that text.
- Calculate the mutual information score for each context word.
- Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.
"""
from pathlib import Path

DATA_DIR = Path("../../CDS-LANG/100_english_novels/corpus")
assert DATA_DIR.exists()

SEARCH_TERM = "input word"
USER_TEXT = ""