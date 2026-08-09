import dataclasses

import pandas as pd
import lexicon_utils
import word_distance

VOWEL_CLUSTER_SET = lexicon_utils.ORTHO_VOWELS | {"'"}

@dataclasses.dataclass
class FormChangeRecord:
    input : str
    input_index : int
    desinence : str
    output : str
    output_index : int
    change_sequences : list[list[tuple[str, str]]]

def allowed_fn(in_str : str, out_str : str) -> bool:
    """Dissalow exchanging vowels for consonants or vice_versa"""
    if not in_str or not out_str: # allow insertions and deletions
        return True
    is_vowel_cluster = [bool(VOWEL_CLUSTER_SET.intersection(s)) for s in (in_str, out_str)]
    return is_vowel_cluster[0] == is_vowel_cluster[1]

def compare_forms(df : pd.DataFrame, index_base : int, index_derived : int) -> FormChangeRecord:
    clusters = [lexicon_utils.index_to_letterclusters(df, i) for i in (index_base, index_derived)]
    clusters[0], desinence = lexicon_utils.generate_root(clusters[0])
    wt = word_distance.WordTransformation(clusters[0], clusters[1], allowed_fn=allowed_fn)
    wt.compute_change_sequences()
    if not wt.change_sequences:
        raise Exception('Could not compute change sequences')
    return FormChangeRecord(input = ''.join(clusters[0]), input_index=index_base,
                            desinence=desinence,
                            output=''.join(clusters[1]), output_index=index_derived,
                            change_sequences=[[(chg.d_in, chg.d_out) for chg in seq] for seq in
                                              wt.change_sequences]
    )


def get_new_ending(chg_seq : list[tuple[str, str]]) -> list[str]:
    chg_seq = chg_seq[:-1] # cut off $
    i = len(chg_seq)-1
    while i >= 0 and chg_seq[i][0] == '':
         i -= 1
    return [t[1] for t in chg_seq[i+1:]]

if __name__ == "__main__":
    import pickle


    df = pd.read_csv('./lexicon/nouns.tsv', sep='\t')
    with open('./lexicon/plurals_indices.p', 'rb') as handle:
        plural_dict : dict[tuple, list] = pickle.load(handle)
    with open('./lexicon/change_records.1.p', 'rb') as handle:
        change_list : list[dict] = pickle.load(handle)
    