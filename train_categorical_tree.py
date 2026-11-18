"""Train and inspect a categorical phonetic-change classifier."""

import argparse
from pathlib import Path

import pandas as pd

from decision_tree import CategoricalDecisionTreeClassifier


def load_data(path, target='result'):
    # Preserve absent desinences and values such as '0' as literal categories.
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    # pandas.to_csv writes an unnamed row index in the supplied dataset.
    if len(df.columns) and df.columns[0] == 'Unnamed: 0':
        df = df.drop(columns=df.columns[0])
    if target not in df.columns:
        raise ValueError(f'Target column {target!r} is missing')
    features = df.drop(columns=target)
    # Record IDs identify source examples; they must not become split features.
    features = features.drop(columns='ID', errors='ignore')
    return features, df[target]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path(__file__).parent / 'train_data/schwa_chg.tsv')
    parser.add_argument('--target', default='result')
    parser.add_argument('--max-depth', type=int)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    args = parser.parse_args()
    X, y = load_data(args.path, args.target)
    model = CategoricalDecisionTreeClassifier(
        max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
    ).fit(X, y)
    print(model.export_text())
    correct = sum(actual == expected for actual, expected in zip(model.predict(X), y))
    print(f'\nTraining accuracy: {correct}/{len(y)} ({model.score(X, y):.2%})')
    print('Training accuracy measures fit to these examples, not accuracy on new data.')


if __name__ == '__main__':
    main()
