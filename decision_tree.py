from typing import Any

import pandas as pd
import dataclasses
from collections import Counter, defaultdict
import itertools

@dataclasses.dataclass
class DecisionTree:
    fname : str
    decisions : dict = dataclasses.field(default_factory=dict)
    outcome : Any = None
    score : float = -1
    keys : list = dataclasses.field(default_factory=list)
    parent : 'DecisionTree|None' = None

    def __str__(self):
        if not self.fname:
            return '-> ' + self.outcome
        return f'{self.fname} -> {list(self.decisions.keys())}'
    def __repr__(self) -> str:
        return str(self)
    
    def __getitem__(self, key):
        return self.decisions[key]
    
def Gini_impurity(df : pd.DataFrame, outcome_name : str) -> float:
    counter = Counter(df[outcome_name])
    probabilities_sq = [(v/len(df))**2 for v in counter.values()]
    return 1 - sum(probabilities_sq)

def generate_split(df : pd.DataFrame, feature_names : list[str], outcome_name : str, key_col : str = '') -> DecisionTree:
    current_score = Gini_impurity(df, outcome_name)
    current_list = df[key_col].to_list() if key_col else []

    outcomes = Counter(df[outcome_name])
    if len(outcomes) < 2 or not feature_names:
        return DecisionTree('', dict(), outcomes.most_common(1)[0][0] if outcomes else '?',
                            current_score, current_list)
    
    splits = []
    for fname in feature_names:
        fvals = set(df[fname])
        # df_dict = {fval:df[df[fname==fval]] for fval in fvals}
        df_dict = {}
        for fval in fvals:
            df_dict[fval] = df[df[fname]==fval]
        scores = [Gini_impurity(df_split, outcome_name) * len(df_split) for fval, df_split in df_dict.items()]
        splits.append({'fname':fname, 'df_split':df_dict, 'score':sum(scores)})
    splits.sort(key=lambda d: d['score'])
    best = splits[0]
    new_features = list(feature_names)
    new_features.remove(best['fname'])
    node = DecisionTree(fname=best['fname'], decisions=
        {fval:generate_split(df_split, new_features, outcome_name, key_col) for fval, df_split in best['df_split'].items()},
        score=current_score, keys=current_list)
    return node

def merge_same_children(node : DecisionTree):
    children_outcomes = defaultdict(list)
    for k, v in node.decisions.items():
        children_outcomes[v.outcome].append((k,v))
    children_outcomes = {k:v for k,v in children_outcomes.items() if k and len(v)>1}
    for outcome, child_list in children_outcomes.items():
        fvals = [c[0] for c in child_list]
        keys = list(itertools.chain.from_iterable([c[1].keys for c in child_list]))
        # remove existing children
        for fval in fvals:
            node.decisions.pop(fval)
        # add new child
        node.decisions[tuple(fvals)] = DecisionTree('', {}, outcome, 0, keys)

def iter_tree(tree : DecisionTree):
    yield tree
    for child in tree.decisions.values():
        for node in iter_tree(child):
            yield node

def get_parent_decisions(node : DecisionTree) -> list[tuple[str, str]]:
    if not node.parent:
        return []
    my_val = [k for k,v in node.parent.decisions.items() if v is node][0]
    return get_parent_decisions(node.parent) + [(node.parent.fname, my_val)]


if __name__ == "__main__":
    df = pd.read_csv('./train_data/a_chg.tsv', sep='\t', encoding='utf-8')
    df = df.fillna('')
    df = df[df['d_in'] != 'EXCL']

    tree = generate_split(df, ['before', 'after', 'desin1', 'desin2'], 'd_out', 'key2')

    for node in iter_tree(tree):
        merge_same_children(node)

    for node in iter_tree(tree):
        for child in node.decisions.values():
            child.parent = node

    leaves = [n for n in iter_tree(tree) if not n.fname]
    leaves = [l for l in leaves if l.outcome != 'a']
    leaves.sort(key=lambda l: -len(l.keys))

