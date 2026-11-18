import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from decision_tree import (
    CategoricalDecisionTreeClassifier,
    generate_split,
    get_parent_decisions,
    iter_tree,
    merge_same_children,
)
from train_categorical_tree import load_data


class CategoricalTreeTests(unittest.TestCase):
    def test_multiway_split_and_column_order(self):
        X = pd.DataFrame({'noise': ['x'] * 6, 'ending': ['i', 'e', 'uri'] * 2})
        y = ["'ă", "'e", "'a"] * 2
        model = CategoricalDecisionTreeClassifier().fit(X, y)
        self.assertEqual(model.tree_.fname, 'ending')
        self.assertEqual(list(model.tree_.decisions), ['i', 'e', 'uri'])
        self.assertEqual(model.predict(X[['ending', 'noise']]), y)
        self.assertEqual(model.score(X, y), 1)

    def test_weighted_impurity_selects_best_feature(self):
        X = pd.DataFrame({'weak': ['a', 'a', 'a', 'b', 'b', 'b'],
                          'strong': ['x', 'x', 'x', 'x', 'y', 'y']})
        model = CategoricalDecisionTreeClassifier().fit(X, [0, 0, 0, 0, 1, 1])
        self.assertEqual(model.tree_.fname, 'strong')

    def test_zero_gain_interaction_is_learned(self):
        X = pd.DataFrame({'a': [0, 0, 1, 1], 'b': [0, 1, 0, 1]})
        y = [0, 1, 1, 0]
        model = CategoricalDecisionTreeClassifier().fit(X, y)
        self.assertEqual(model.predict(X), y)
        self.assertEqual(model.tree_.fname, 'a')  # Feature-order tie break.
        limited = CategoricalDecisionTreeClassifier(min_impurity_decrease=0.01).fit(X, y)
        self.assertFalse(limited.tree_.fname)

    def test_conflicting_context_retains_probabilities(self):
        X = pd.DataFrame({'before': ['l'] * 3})
        model = CategoricalDecisionTreeClassifier().fit(X, ["'ă", "'a", "'ă"])
        self.assertEqual(model.predict(X), ["'ă"] * 3)
        self.assertEqual(model.classes_, ["'ă", "'a"])
        self.assertEqual(model.predict_proba(X)[0], [2 / 3, 1 / 3])
        tied = CategoricalDecisionTreeClassifier().fit(X.iloc[:2], ["'a", "'ă"])
        self.assertEqual(tied.predict(X.iloc[:1]), ["'a"])

    def test_unseen_value_uses_nearest_node_distribution(self):
        X = pd.DataFrame({'a': ['x', 'x', 'x', 'y', 'y', 'y', 'y'],
                          'b': ['p', 'p', 'q', 'p', 'p', 'q', 'q']})
        model = CategoricalDecisionTreeClassifier().fit(X, [0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(model.tree_.fname, 'a')
        query = pd.DataFrame({'a': ['x', 'unseen'], 'b': ['unseen', 'p']})
        self.assertEqual(model.predict(query), [0, 1])
        self.assertEqual(model.predict_proba(query), [[2 / 3, 1 / 3], [2 / 7, 5 / 7]])

    def test_empty_and_missing_values_are_one_category(self):
        X = pd.DataFrame({'ending': pd.Series(['', pd.NA, 'i', 'i'], dtype='string')})
        original = X.copy(deep=True)
        model = CategoricalDecisionTreeClassifier().fit(X, ['absent', 'absent', 'i', 'i'])
        query = pd.DataFrame({'ending': [None, float('nan'), '', 'i']})
        self.assertEqual(model.predict(query), ['absent', 'absent', 'absent', 'i'])
        pd.testing.assert_frame_equal(X, original)

    def test_stopping_controls(self):
        X = pd.DataFrame({'ending': ['i', 'i', 'e']})
        for options in ({'max_depth': 0}, {'min_samples_leaf': 2},
                        {'min_impurity_decrease': 0.5}):
            with self.subTest(options=options):
                model = CategoricalDecisionTreeClassifier(**options).fit(X, ['a', 'a', 'e'])
                self.assertFalse(model.tree_.fname)
                self.assertEqual(model.predict(X), ['a'] * 3)

    def test_no_features_and_empty_predictions(self):
        X = pd.DataFrame(index=range(3))
        model = CategoricalDecisionTreeClassifier().fit(X, ['a', 'e', 'a'])
        self.assertEqual(model.predict(X), ['a'] * 3)
        self.assertEqual(model.predict_proba(X), [[2 / 3, 1 / 3]] * 3)
        self.assertEqual(model.predict(X.iloc[:0]), [])

    def test_refit_replaces_classes_and_tree(self):
        X = pd.DataFrame({'ending': ['i', 'e']}, index=[20, 10])
        model = CategoricalDecisionTreeClassifier().fit(X, pd.Series(['a', 'e']))
        self.assertEqual(model.predict(X), ['a', 'e'])
        model.fit(X, ['new', 'new'])
        self.assertEqual(model.classes_, ['new'])
        self.assertEqual(model.predict_proba(X), [[1], [1]])

    def test_sample_keys_preserve_record_ids_and_conditions(self):
        X = pd.DataFrame({'ending': ['i', 'e', 'i']}, index=[30, 20, 10])
        model = CategoricalDecisionTreeClassifier().fit(
            X, ['ă', 'e', 'ă'], sample_keys=pd.Series([384, 385, 490]),
        )
        leaf = model.tree_['i']
        self.assertEqual(model.tree_.keys, [384, 385, 490])
        self.assertEqual(leaf.keys, [384, 490])
        self.assertEqual(leaf.keys_good, [384, 490])
        self.assertEqual(leaf.keys_bad, [])
        self.assertEqual(get_parent_decisions(leaf), [('ending', 'i')])
        self.assertEqual(model.feature_names_in_, ['ending'])
        self.assertEqual(model.predict(X), ['ă', 'e', 'ă'])
        with self.assertRaisesRegex(ValueError, 'sample_keys'):
            model.fit(X, ['ă', 'e', 'ă'], sample_keys=[384])
        model.fit(X, ['ă', 'e', 'ă'])
        self.assertEqual(model.tree_.keys, [])
        self.assertEqual(model.tree_.keys_good, [])
        self.assertEqual(model.tree_.keys_bad, [])

    def test_matching_and_exception_keys_for_conflicting_and_stopped_leaves(self):
        X = pd.DataFrame({'before': ['l', 'l', 'l']}, index=[30, 20, 10])
        model = CategoricalDecisionTreeClassifier().fit(
            X, ["'ă", "'a", "'ă"], sample_keys=[384, 385, 490],
        )
        self.assertEqual(model.tree_.outcome, "'ă")
        self.assertEqual(model.tree_.keys_good, [384, 490])
        self.assertEqual(model.tree_.keys_bad, [385])
        tied = CategoricalDecisionTreeClassifier().fit(
            X.iloc[:2], ["'a", "'ă"], sample_keys=[384, 385],
        )
        self.assertEqual(tied.tree_.keys_good, [384])
        self.assertEqual(tied.tree_.keys_bad, [385])
        distinct = pd.DataFrame({'before': ['l', 'm', 'n']})
        for options in ({'max_depth': 0}, {'min_samples_leaf': 2},
                        {'min_impurity_decrease': 0.5}):
            with self.subTest(options=options):
                stopped = CategoricalDecisionTreeClassifier(**options).fit(
                    distinct, ["'ă", "'a", "'ă"], sample_keys=[384, 385, 490],
                )
                self.assertFalse(stopped.tree_.fname)
                self.assertEqual(stopped.tree_.keys_good, [384, 490])
                self.assertEqual(stopped.tree_.keys_bad, [385])

    def test_merging_impure_leaves_preserves_key_groups_and_duplicates(self):
        df = pd.DataFrame({'ending': ['a', 'a', 'a', 'b', 'b', 'b', 'c'],
                           'result': [0, 1, 0, 0, 0, 1, 1],
                           'key': [10, 11, 10, 13, 14, 15, 16]})
        tree = generate_split(df, ['ending'], 'result', 'key')
        self.assertEqual(tree['a'].keys_good, [10, 10])
        self.assertEqual(tree['a'].keys_bad, [11])
        for _ in range(2):
            merge_same_children(tree)
            merged = tree[('a', 'b')]
            self.assertEqual(merged.outcome, 0)
            self.assertEqual(merged.keys, [10, 11, 10, 13, 14, 15])
            self.assertEqual(merged.keys_good, [10, 10, 13, 14])
            self.assertEqual(merged.keys_bad, [11, 15])
            self.assertEqual(merged.class_counts, {0: 4, 1: 2})
            self.assertEqual(tree['c'].keys_good, [16])
            self.assertEqual(tree['c'].keys_bad, [])

    def test_invalid_inputs(self):
        X = pd.DataFrame({'a': ['x', 'y']})
        for options in ({'max_depth': -1}, {'max_depth': 1.5},
                        {'min_samples_leaf': 0}, {'min_impurity_decrease': math.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CategoricalDecisionTreeClassifier(**options)
        model = CategoricalDecisionTreeClassifier()
        for data, labels in ((X.iloc[:0], []), (X, [0]), (X, [0, None])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                model.fit(data, labels)
        with self.assertRaises(ValueError):
            model.predict(X)
        model.fit(X, [0, 1])
        with self.assertRaisesRegex(ValueError, 'Missing feature'):
            model.predict(pd.DataFrame({'other': ['x']}))
        with self.assertRaises(ValueError):
            model.fit(pd.DataFrame([['x', 'y']], columns=['a', 'a']), [0])

    def test_legacy_tree_helpers_and_merging(self):
        df = pd.DataFrame({'ending': ['a', 'b', 'c'], 'result': [0, 0, 1], 'key': [1, 2, 3]})
        tree = generate_split(df, ['ending'], 'result', 'key')
        self.assertEqual(get_parent_decisions(tree['a']), [('ending', 'a')])
        merge_same_children(tree)
        merged = tree[('a', 'b')]
        self.assertEqual(merged.keys, [1, 2])
        self.assertEqual(merged.class_counts, {0: 2})
        self.assertIs(merged.parent, tree)
        self.assertEqual(len(list(iter_tree(tree))), 3)
        model = CategoricalDecisionTreeClassifier().fit(df[['ending']], df.result)
        merge_same_children(model.tree_)
        self.assertEqual(model.predict(df[['ending']]), [0, 0, 1])
        self.assertEqual(model.predict_proba(df[['ending']]), [[1, 0], [1, 0], [0, 1]])

    def test_rules_are_readable_and_deterministic(self):
        X = pd.DataFrame({'ending': ['i', 'e', 'uri']})
        first = CategoricalDecisionTreeClassifier().fit(X, ["'ă", "'e", "'ă"])
        second = CategoricalDecisionTreeClassifier().fit(X, ["'ă", "'e", "'ă"])
        self.assertEqual(first.export_text(), second.export_text())
        self.assertIn("== 'i':", first.export_text())
        self.assertIn('n=3', first.export_text())

    def test_tsv_loader_with_and_without_saved_index(self):
        for header, row in (('\tending\tresult', '0\t\tă'),
                            ('ending\tresult', '\tă'),
                            ('ID\tending\tresult', '384\t\tă')):
            with self.subTest(header=header), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'data.tsv'
                path.write_text(header + '\n' + row + '\n', encoding='utf-8')
                X, y = load_data(path)
                self.assertEqual(X.columns.tolist(), ['ending'])
                self.assertEqual(X.iloc[0, 0], '')
                self.assertEqual(y.tolist(), ['ă'])

    def test_schwa_training_reaches_conflicting_context_ceiling(self):
        path = Path(__file__).parent / 'train_data/schwa_chg.tsv'
        if not path.exists():
            self.skipTest('Local generated dataset is not available')
        X, y = load_data(path)
        model = CategoricalDecisionTreeClassifier().fit(X, y)
        self.assertNotIn('Unnamed: 0', model.feature_names_in_)
        self.assertNotIn('ID', model.feature_names_in_)
        self.assertEqual(sum(a == b for a, b in zip(model.predict(X), y)), 51)
        probabilities = model.predict_proba(X)
        self.assertEqual(probabilities[0], probabilities[36])
        self.assertEqual(probabilities[0][model.classes_.index("'a")], 0.5)
        self.assertEqual(probabilities[0][model.classes_.index("'ă")], 0.5)


if __name__ == '__main__':
    unittest.main()
