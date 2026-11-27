"""Focused checks for choosing categorical splits in a supplied feature order."""

import unittest

import pandas as pd

from decision_tree import (
    CategoricalDecisionTreeClassifier,
    generate_split,
    get_parent_decisions,
    iter_tree,
    merge_same_children,
)


class FeatureOrderTests(unittest.TestCase):
    def setUp(self):
        # 'later' predicts perfectly; 'early' has no immediate impurity gain.
        self.X = pd.DataFrame({'early': ['x', 'x', 'y', 'y'],
                               'later': ['p', 'q', 'p', 'q']})
        self.y = [0, 1, 0, 1]

    def test_forced_order_overrides_scores_and_follows_column_reordering(self):
        default = CategoricalDecisionTreeClassifier().fit(self.X, self.y)
        self.assertEqual(default.tree_.fname, 'later')
        ordered = CategoricalDecisionTreeClassifier(force_feature_order=True).fit(self.X, self.y)
        self.assertEqual(ordered.tree_.fname, 'early')
        self.assertEqual(ordered.tree_['x'].fname, 'later')
        self.assertEqual(ordered.tree_['y'].fname, 'later')
        self.assertEqual(ordered.predict(self.X), self.y)
        self.assertEqual(ordered.predict_proba(self.X), [[1, 0], [0, 1], [1, 0], [0, 1]])
        ordered.fit(self.X[['later', 'early']], self.y)
        self.assertEqual(ordered.tree_.fname, 'later')
        self.assertTrue(all(not child.fname for child in ordered.tree_.decisions.values()))

    def test_constants_are_skipped_independently_within_each_branch(self):
        X = pd.DataFrame({'constant': ['same'] * 6,
                          'a': ['x', 'x', 'y', 'y', 'y', 'y'],
                          'b': ['p', 'p', 'p', 'p', 'q', 'q'],
                          'c': ['m', 'n', 'm', 'n', 'm', 'n']})
        model = CategoricalDecisionTreeClassifier(force_feature_order=True).fit(X, [0, 1, 0, 1, 1, 0])
        self.assertEqual(model.tree_.fname, 'a')
        self.assertEqual(model.tree_['x'].fname, 'c')
        self.assertEqual(model.tree_['y'].fname, 'b')
        self.assertEqual(model.tree_['y']['p'].fname, 'c')
        self.assertEqual(model.predict(X), [0, 1, 0, 1, 1, 0])
        for node in iter_tree(model.tree_):
            positions = [list(X.columns).index(feature)
                         for feature, _ in get_parent_decisions(node)]
            self.assertEqual(positions, sorted(set(positions)))

    def test_pure_and_featureless_nodes_still_stop(self):
        for X in (self.X, pd.DataFrame(index=range(4))):
            with self.subTest(columns=list(X.columns)):
                model = CategoricalDecisionTreeClassifier(force_feature_order=True).fit(X, [1] * 4)
                self.assertFalse(model.tree_.fname)
                self.assertEqual(model.predict(X), [1] * 4)
        constant = CategoricalDecisionTreeClassifier(force_feature_order=True).fit(
            pd.DataFrame({'a': ['x'] * 4}), self.y,
        )
        self.assertFalse(constant.tree_.fname)

    def test_depth_and_impurity_limits_are_not_overridden(self):
        for options in ({'max_depth': 0}, {'min_impurity_decrease': 0.1}):
            with self.subTest(options=options):
                model = CategoricalDecisionTreeClassifier(
                    force_feature_order=True, **options,
                ).fit(self.X, self.y)
                self.assertFalse(model.tree_.fname)
        limited = CategoricalDecisionTreeClassifier(
            force_feature_order=True, max_depth=1,
        ).fit(self.X, self.y)
        self.assertEqual(limited.tree_.fname, 'early')
        self.assertTrue(all(not child.fname for child in limited.tree_.decisions.values()))

    def test_leaf_size_limit_stops_instead_of_selecting_a_later_feature(self):
        X = pd.DataFrame({'early': ['x', 'x', 'x', 'y'],
                          'later': ['p', 'p', 'q', 'q']})
        y = [0, 0, 1, 1]
        default = CategoricalDecisionTreeClassifier(min_samples_leaf=2).fit(X, y)
        self.assertEqual(default.tree_.fname, 'later')
        ordered = CategoricalDecisionTreeClassifier(
            force_feature_order=True, min_samples_leaf=2,
        ).fit(X, y)
        self.assertFalse(ordered.tree_.fname)

    def test_direct_builder_and_merged_exception_keys(self):
        df = self.X.assign(result=self.y, ID=[10, 11, 12, 13])
        tree = generate_split(df, ['early', 'later'], 'result', 'ID',
                              force_feature_order=True, max_depth=1)
        self.assertEqual(tree.fname, 'early')
        self.assertEqual(tree['x'].keys_good, [10])
        self.assertEqual(tree['x'].keys_bad, [11])
        merge_same_children(tree)
        merged = tree[('x', 'y')]
        self.assertEqual(merged.keys, [10, 11, 12, 13])
        self.assertEqual(merged.keys_good, [10, 12])
        self.assertEqual(merged.keys_bad, [11, 13])
        self.assertEqual(get_parent_decisions(merged), [('early', ('x', 'y'))])


if __name__ == '__main__':
    unittest.main()
