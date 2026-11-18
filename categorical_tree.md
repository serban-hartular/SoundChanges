The categorical decision tree
=============================

`decision_tree.py` provides `CategoricalDecisionTreeClassifier`. It splits
directly on feature values, such as `desinence_out == 'ie'`, with one branch
per observed category. It needs pandas, but no sklearn or categorical encoder.
The existing `generate_split` and tree inspection helpers remain available.

Run the example on the schwa dataset:

```sh
.venv/bin/python train_categorical_tree.py
.venv/bin/python train_categorical_tree.py train_data/schwa_chg.tsv --max-depth 3
```

Use it in Python:

```python
from decision_tree import CategoricalDecisionTreeClassifier
from train_categorical_tree import load_data

X, y = load_data('train_data/schwa_chg.tsv')
model = CategoricalDecisionTreeClassifier().fit(X, y)
print(model.export_text())

# Predict a new context. Use the same value types as in training.
context = X.iloc[[0]].copy()
context['before'] = 'd'
print(model.predict(context))
print(model.classes_)                 # Column order for predict_proba
print(model.predict_proba(context))
```

`load_data` reads all values as strings, preserves empty endings as `''`, and
removes the unnamed row index written by pandas and the `ID` column. For other datasets, pass
your feature DataFrame and target Series directly to `fit`. Their rows are
paired by position. Prediction accepts a DataFrame and matches columns by
name; additional columns are ignored and required columns must be present.
`predict` and `predict_proba` return Python lists.

To explore individual rules, use the classifier's `tree_` and the tree helpers:

```python
from decision_tree import iter_tree, get_parent_decisions

leaves = [node for node in iter_tree(model.tree_) if not node.fname]
for leaf in leaves:
    print(get_parent_decisions(leaf), '=>', leaf.outcome, leaf.class_counts)
```

The conditions on each path are joined with AND. A tuple of values in a
condition represents membership after calling `merge_same_children`.
To track source records, pass `sample_keys=df['ID']` to `fit`; each node's
`keys` then contains its training record IDs. These IDs are metadata, not
features. The main block in `decision_tree.py` stores each classifier and
its leaves in `tree_dict`, indexed by the desinence pair.

Each node also has `keys_good` (IDs whose observed result matches its
`outcome`) and `keys_bad` (IDs whose result differs). These lists are populated
when passing `sample_keys` to `fit`, or `key_col` to `generate_split`, and are
preserved when merging sibling leaves. Without record keys, all three key
lists are empty. For example, to view exceptions to a leaf's rule:

```python
leaf = leaves[0]
orig_df.loc[orig_df['ID'].isin(leaf.keys_bad), ['input_form', 'output_form']]
```

The tree greedily minimizes weighted Gini impurity. Feature order breaks ties
between equally good splits; the first observed label within a node breaks
ties between majority classes. Constant features are skipped. Each feature
can occur only once along a path. Zero-gain splits are allowed by default
because they can uncover interactions, such as XOR, at later levels.

Missing feature values (`None`, NaN, `pd.NA`) become `''` during training and
prediction, matching absent endings in this dataset. Labels cannot be missing.
Numbers, including `is_last_syll`, are treated as categories, not thresholds;
use consistent types when training and predicting (`'1'` and `1` differ).

An unseen value stops traversal at the current node and uses that node's
majority class. `predict_proba` returns its observed class frequencies. These
are empirical frequencies, not calibrated confidence estimates.

Identical feature rows can have different labels. The supplied dataset has
one such context: `(u, ie, l, u, 1)` occurs with both `'a` and `'ă`. Its leaf
keeps both counts, predicts the first observed label in the tie, and gives
each result probability 0.5. Consequently the unrestricted tree fits 51/52
training rows (98.08%). This measures training fit; assess predictions on new
data using held-out examples.

To limit tree size, set `max_depth` (root depth is 0) or `min_samples_leaf`
(every branch of a proposed split must have enough examples). A positive
`min_impurity_decrease` requires that much local reduction in Gini impurity.
Multiway categorical splits can favor features with many distinct values;
exclude identifiers and use these controls when generalization matters.

Run the focused tests with:

```sh
.venv/bin/python -m unittest test_categorical_tree -v
```
