"""Multiway decision trees that split directly on categorical feature values."""

from collections import Counter, defaultdict
import dataclasses
import itertools
import math
from numbers import Integral
from typing import Any

import pandas as pd


@dataclasses.dataclass
class DecisionTree:
    fname: str
    decisions: dict = dataclasses.field(default_factory=dict)
    outcome: Any = None
    score: float = -1
    keys: list = dataclasses.field(default_factory=list)
    parent: 'DecisionTree | None' = dataclasses.field(default=None, repr=False, compare=False)
    class_counts: dict = dataclasses.field(default_factory=dict)
    keys_good: list = dataclasses.field(default_factory=list)
    keys_bad: list = dataclasses.field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return sum(self.class_counts.values())

    def __str__(self):
        if not self.fname:
            return '-> ' + str(self.outcome)
        return f'{self.fname} -> {list(self.decisions.keys())}'

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key):
        return self.decisions[key]


def Gini_impurity(df: pd.DataFrame, outcome_name: str) -> float:
    if df.empty:
        return 0.0
    counter = Counter(df[outcome_name])
    return 1 - sum((count / len(df)) ** 2 for count in counter.values())


def _validate_options(max_depth, min_samples_leaf, min_impurity_decrease):
    if max_depth is not None and (not isinstance(max_depth, Integral) or max_depth < 0):
        raise ValueError('max_depth must be a nonnegative integer or None')
    if not isinstance(min_samples_leaf, Integral) or min_samples_leaf < 1:
        raise ValueError('min_samples_leaf must be a positive integer')
    if not math.isfinite(min_impurity_decrease) or min_impurity_decrease < 0:
        raise ValueError('min_impurity_decrease must be finite and nonnegative')


def _validate_features(df, feature_names):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Features must be a pandas DataFrame')
    if not df.columns.is_unique or len(set(feature_names)) != len(feature_names):
        raise ValueError('Column and feature names must be unique')
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise ValueError('Feature names must be nonempty strings')
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f'Missing feature columns: {missing}')


def _normalize_features(df):
    # Empty strings represent absent desinences in the phonetic datasets.
    return df.astype(object).where(df.notna(), '')


def generate_split(
    df: pd.DataFrame, feature_names: list[str], outcome_name: str, key_col: str = '',
    *, max_depth: int | None = None, min_samples_leaf: int = 1,
    min_impurity_decrease: float = 0.0,
) -> DecisionTree:
    """Build a categorical tree using weighted Gini impurity.

    Each split has one branch per observed value; a feature is used at most
    once per path. Ties favor feature order, then first-observed class order
    within a node. Zero-gain splits are allowed by default, so interactions
    such as XOR can be learned. Missing feature values become ''.
    """
    _validate_options(max_depth, min_samples_leaf, min_impurity_decrease)
    feature_names = list(feature_names)
    _validate_features(df, feature_names)
    if df.empty:
        raise ValueError('Cannot train on an empty dataset')
    if outcome_name in feature_names:
        raise ValueError('The outcome column cannot also be a feature')
    if df[outcome_name].isna().any():
        raise ValueError('Outcome values cannot be missing')
    data = df.copy()
    data[feature_names] = _normalize_features(data[feature_names])

    def build(rows, features, depth, parent=None):
        outcomes = Counter(rows[outcome_name])
        node = DecisionTree(
            fname='', outcome=outcomes.most_common(1)[0][0],
            score=Gini_impurity(rows, outcome_name),
            keys=rows[key_col].tolist() if key_col else [],
            parent=parent, class_counts=dict(outcomes),
        )
        if key_col:
            matches = rows[outcome_name] == node.outcome
            node.keys_good = rows.loc[matches, key_col].tolist()
            node.keys_bad = rows.loc[~matches, key_col].tolist()
        if len(outcomes) == 1 or not features or (max_depth is not None and depth >= max_depth):
            return node

        best_feature, best_groups, best_score = None, None, math.inf
        for feature in features:
            groups = list(rows.groupby(feature, sort=False, observed=True, dropna=False))
            if len(groups) < 2 or any(len(group) < min_samples_leaf for _, group in groups):
                continue
            score = sum(len(group) * Gini_impurity(group, outcome_name)
                        for _, group in groups) / len(rows)
            if score < best_score - 1e-12:
                best_feature, best_groups, best_score = feature, groups, score

        # The threshold applies to the local reduction in Gini impurity.
        if best_feature is None or node.score - best_score + 1e-12 < min_impurity_decrease:
            return node
        node.fname = best_feature
        remaining = [feature for feature in features if feature != best_feature]
        node.decisions = {value: build(group, remaining, depth + 1, node)
                          for value, group in best_groups}
        return node

    return build(data, feature_names, 0)


def merge_same_children(node: DecisionTree):
    """Group sibling leaves with the same prediction, retaining counts and key groups."""
    children_outcomes = defaultdict(list)
    for key, child in node.decisions.items():
        if not child.fname:
            children_outcomes[child.outcome].append((key, child))
    for outcome, child_list in children_outcomes.items():
        if len(child_list) < 2:
            continue
        old_keys = [key for key, _ in child_list]
        values = list(itertools.chain.from_iterable(
            key if isinstance(key, tuple) else (key,) for key in old_keys))
        keys = list(itertools.chain.from_iterable(child.keys for _, child in child_list))
        counts = Counter()
        for _, child in child_list:
            counts.update(child.class_counts)
        for key in old_keys:
            node.decisions.pop(key)
        total = sum(counts.values())
        score = 1 - sum((count / total) ** 2 for count in counts.values()) if total else 0
        node.decisions[tuple(values)] = DecisionTree(
            '', {}, outcome, score, keys, node, dict(counts),
            keys_good=list(itertools.chain.from_iterable(
                child.keys_good for _, child in child_list)),
            keys_bad=list(itertools.chain.from_iterable(
                child.keys_bad for _, child in child_list)),
        )


def iter_tree(tree: DecisionTree):
    yield tree
    for child in tree.decisions.values():
        yield from iter_tree(child)


def get_parent_decisions(node: DecisionTree) -> list[tuple[str, Any]]:
    if node.parent is None:
        return []
    value = next(key for key, child in node.parent.decisions.items() if child is node)
    return get_parent_decisions(node.parent) + [(node.parent.fname, value)]


class CategoricalDecisionTreeClassifier:
    """Classifier for discrete features, without encoding or numeric thresholds.

    Numeric values are categories too. Unknown values stop traversal at the
    current node, whose majority class and distribution are returned. Missing
    values are treated as ''. Conflicting rows retain their class distribution.
    Input values and labels must be hashable scalars, such as strings or ints.
    """

    def __init__(self, *, max_depth=None, min_samples_leaf=1, min_impurity_decrease=0.0):
        _validate_options(max_depth, min_samples_leaf, min_impurity_decrease)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X: pd.DataFrame, y, *, sample_keys=None):
        """Fit X to y in row order; pandas indices are ignored.

        Optional sample_keys attach record IDs to each node's keys without
        using those IDs as predictive features. keys_good contains IDs whose
        labels match that node's outcome; keys_bad contains the exceptions.
        """
        _validate_features(X, list(X.columns) if isinstance(X, pd.DataFrame) else [])
        labels = list(y)
        if len(labels) != len(X):
            raise ValueError('X and y must contain the same number of rows')
        target = '__target__'
        while target in X.columns:
            target += '_'
        data = X.copy()
        data[target] = labels
        key_col = ''
        if sample_keys is not None:
            keys = list(sample_keys)
            if len(keys) != len(X):
                raise ValueError('X and sample_keys must contain the same number of rows')
            key_col = '__sample_key__'
            while key_col in data.columns:
                key_col += '_'
            data[key_col] = keys
        tree = generate_split(
            data, list(X.columns), target, key_col, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )
        self.tree_ = tree
        self.feature_names_in_ = list(X.columns)
        self.classes_ = list(dict.fromkeys(labels))
        return self

    def _check_fitted(self):
        if not hasattr(self, 'tree_'):
            raise ValueError('Call fit before predicting or exporting the tree')

    def _predict_nodes(self, X):
        self._check_fitted()
        _validate_features(X, self.feature_names_in_)
        if not self.feature_names_in_:
            return [self.tree_] * len(X)
        records = _normalize_features(X[self.feature_names_in_]).to_dict(orient='records')
        nodes = []
        for record in records:
            node = self.tree_
            while node.fname:
                value = record[node.fname]
                child = node.decisions.get(value)
                if child is None:
                    # Support trees simplified with merge_same_children().
                    child = next((branch for values, branch in node.decisions.items()
                                  if isinstance(values, tuple) and value in values), None)
                if child is None:
                    break
                node = child
            nodes.append(node)
        return nodes

    def predict(self, X: pd.DataFrame) -> list:
        return [node.outcome for node in self._predict_nodes(X)]

    def predict_proba(self, X: pd.DataFrame) -> list[list[float]]:
        """Empirical probabilities, with columns ordered as classes_."""
        return [[node.class_counts.get(label, 0) / node.n_samples for label in self.classes_]
                for node in self._predict_nodes(X)]

    def score(self, X: pd.DataFrame, y) -> float:
        """Accuracy; use held-out examples to estimate generalization."""
        predicted, labels = self.predict(X), list(y)
        if not labels or len(labels) != len(predicted):
            raise ValueError('Scoring requires equally sized, nonempty X and y')
        return sum(actual == expected for actual, expected in zip(predicted, labels)) / len(labels)

    def export_text(self) -> str:
        """Show categorical branches, majority predictions and class counts."""
        self._check_fitted()
        lines = []

        def walk(node, indent):
            summary = (f'predict {node.outcome!r} '
                       f'(n={node.n_samples}, counts={node.class_counts!r})')
            if not node.fname:
                lines.append(indent + summary)
                return
            lines.append(indent + f'{node.fname} [fallback: {summary}]')
            for value, child in node.decisions.items():
                operator = 'in' if isinstance(value, tuple) else '=='
                lines.append(indent + f'  {operator} {value!r}:')
                walk(child, indent + '    ')

        walk(self.tree_, '')
        return '\n'.join(lines)


if __name__ == '__main__':
    orig_df = pd.read_csv('./lang_data/change_records.doom.v2.work2.csv', sep='\t').fillna('')
    df = pd.read_csv('./train_data/yback_chg.tsv', sep='\t', encoding='utf-8').fillna('')

    def display_wordpairs(id_list: list[int], limit=3):
        print(orig_df[orig_df['ID'].isin(id_list)][['ID', 'input_form', 'output_form']][:limit].to_string(index=False, header=False))

    tree_dict = {}

    # get desinence pairs
    desinence_cols = ['desinence_in', 'desinence_out']
    desinence_pairs = set(df[desinence_cols].itertuples(index=False, name=None))
    NO_CHG = "'â"
    for des_pair in desinence_pairs:
        df_des = df[(df['desinence_in']==des_pair[0]) & (df['desinence_out']==des_pair[1])]
        
        result_set = set(df_des['result'])
        if len(result_set) == 1:
            result = result_set.pop()
            if result != NO_CHG:
                print(des_pair, result)
                display_wordpairs(list(df_des['ID']))
                print()
        else:
            # ID	desinence_in	desinence_out	before	after	is_last_syll	result
            X_cols = ['is_last_syll','before', 'after',]
            y_col = 'result'
            ID_col = 'ID'
            classifier = CategoricalDecisionTreeClassifier().fit(
                df_des[X_cols], df_des[y_col], sample_keys=df_des[ID_col],
            )
            tree = classifier.tree_
            for node in iter_tree(tree):
                merge_same_children(node)
            leaves = [node for node in iter_tree(tree) if not node.fname and node.outcome != 'a']
            leaves.sort(key=lambda leaf: -len(leaf.keys))
            if len(leaves) == 1: # just one:
                print(des_pair, '->', leaves[0].outcome, 'with exceptions')
                display_wordpairs(leaves[0].keys_good)
                print('\tException(s):')
                display_wordpairs(leaves[0].keys_bad)
                continue
            print(des_pair)
            for leaf in leaves:
                print(get_parent_decisions(leaf), '->', leaf.outcome)
                display_wordpairs(leaf.keys_good)
                if leaf.keys_bad:
                    print('\tException(s):')
                    display_wordpairs(leaf.keys_bad)
            print()
            

    # rules = [
    #     {'conditions': get_parent_decisions(leaf), 'prediction': leaf.outcome,
    #      'class_counts': leaf.class_counts, 'keys': leaf.keys}
    #     for leaf in leaves
    # ]
    # print(classifier.export_text())
