import src.augment as aug
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def test_synonym_augment():
    df = pd.DataFrame(
        {
            "label": [0, 1],
            "text": ["I love you", "I hate you"],
        }
    )
    aug_df = aug.synonym_augment(df, "text", "label", n=3)
    logging.info(aug_df)
    assert type(aug_df) == pd.DataFrame
    assert aug_df.loc[0, "label"] == 0
    assert aug_df.columns.tolist() == ["label", "text"]
    assert aug_df.shape[0] == 6
