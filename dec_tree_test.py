import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv('./train_data/train_data_Aa.tsv', sep='\t')
df = df.fillna('')
X, y = (df[['immediate_back_in', 'consonants_back_in', 'vowels_back_in',
              'immediate_forward_in', 'consonants_forward_in', 'vowels_forward_in',
              'immediate_back_out', 'consonants_back_out', 'vowels_back_out',
              'immediate_forward_out', 'consonants_forward_out', 'vowels_forward_out']],
        df['ph_changed'])
encoder = OneHotEncoder(handle_unknown="ignore")

tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5
)

model = make_pipeline(encoder, tree)
model.fit(X, y)

import numpy as np
import pandas as pd
from sklearn.tree import _tree

feature_names = encoder.get_feature_names_out(X.columns)

def extract_leaf_rules(tree, feature_names):
    t = tree.tree_
    rules = []

    def walk(node, conditions):
        # Leaf
        if t.feature[node] == _tree.TREE_UNDEFINED:
            predicted_class_index = np.argmax(t.value[node][0])
            predicted_class = tree.classes_[predicted_class_index]

            rules.append({
                "leaf_id": node,
                "conditions": conditions.copy(),
                "predicted_class": predicted_class,
            })
            return

        feature = feature_names[t.feature[node]]
        threshold = t.threshold[node]

        # left branch: <= threshold
        walk(
            t.children_left[node],
            conditions + [(feature, "<=", threshold)]
        )

        # right branch: > threshold
        walk(
            t.children_right[node],
            conditions + [(feature, ">", threshold)]
        )

    walk(0, [])
    return rules

def simplify_conditions(conditions, possible_values):
    allowed = {
        feature: set(values)
        for feature, values in possible_values.items()
    }

    # Longest feature names first, in case one feature name
    # happens to be a prefix of another.
    features = sorted(possible_values, key=len, reverse=True)

    for encoded_feature, op, threshold in conditions:

        # Find which original feature this one-hot column belongs to
        feature = next(
            (f for f in features if encoded_feature.startswith(f + "_")),
            None
        )

        if feature is None:
            raise ValueError(
                f"Cannot identify original feature for {encoded_feature!r}"
            )

        value = encoded_feature[len(feature) + 1:]

        # OneHotEncoder produces 0/1 columns, so a split at 0.5 means:
        #
        #   > 0.5   -> feature == value
        #   <= 0.5  -> feature != value

        if op == ">":
            allowed[feature] &= {value}
        elif op == "<=":
            allowed[feature].discard(value)
        else:
            raise ValueError(f"Unexpected operator: {op}")

    # Only show features actually constrained by the rule
    constrained = {
        feature: values
        for feature, values in allowed.items()
        if values != set(possible_values[feature])
    }

    return constrained
