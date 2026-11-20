import dataclasses
import itertools

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

def compare_entries(e0 : lexicon_utils.FormEntry, e1 : lexicon_utils.FormEntry) -> FormChangeRecord:
    clusters = [lexicon_utils.entry_to_letterclusters(e) for e in (e0, e1)]
    clusters[0], desinence = lexicon_utils.generate_root(clusters[0])
    wt = word_distance.WordTransformation(clusters[0], clusters[1], allowed_fn=allowed_fn)
    wt.compute_change_sequences()
    if not wt.change_sequences:
        raise Exception('Could not compute change sequences')
    return FormChangeRecord(input = ''.join(clusters[0]), input_index=-1,
                            desinence=desinence,
                            output=''.join(clusters[1]), output_index=-1,
                            change_sequences=[[(chg.d_in, chg.d_out) for chg in seq] for seq in
                                              wt.change_sequences]
    )



def get_new_ending(chg_seq : list[tuple[str, str]]) -> list[str]:
    chg_seq = chg_seq[:-1] # cut off $
    i = len(chg_seq)-1
    while i >= 0 and chg_seq[i][0] == '':
         i -= 1
    return [t[1] for t in chg_seq[i+1:]]


def form_ends_with(form:str, end:str) -> bool:
    if form and form[-1] not in lexicon_utils.ORTHO_VOWELS:
        form = '0'
    return form.endswith(end)
 
def noun_matches_endings(form_sg:str, form_pl:str, ends_sg:list[str], ends_pl:list[str]) -> list[tuple]:
    matches = []
    for e_sg, e_pl in zip(ends_sg, ends_pl):
        if form_ends_with(form_sg, e_sg) and form_ends_with(form_pl, e_pl):
            matches.append((e_sg, e_pl))
    return matches

if __name__ == "__main__":
    import pickle

    print('Loading')
    df = pd.read_csv('./lexicon/nouns.v3.tsv', sep='\t')
    with open('./lexicon/plurals_indices.p', 'rb') as handle:
        plural_dict : dict[tuple, list] = pickle.load(handle)

    flex_df = pd.read_csv('./lang_data/subst_flex_clase.tsv', sep='\t')

    print('Extracting data')
    plural_dict = {k:v for k,v in plural_dict.items() if v}
    index_set = set([t[0] for t in plural_dict.keys()]) |\
            set(itertools.chain.from_iterable([[t[0] for t in v] for v in plural_dict.values()]))
    entries = df[df['INDEX'].isin(index_set)].to_dict(orient='records')
    entry_dict = {d['INDEX']:lexicon_utils.FormEntry.from_dict(d) for d in entries}
    

    def generate_change_records():
        print('Loading')
        df = pd.read_csv('./lexicon/nouns.v3.tsv', sep='\t')
        with open('./lexicon/plurals_indices.p', 'rb') as handle:
            plural_dict : dict[tuple, list] = pickle.load(handle)

        print('Extracting data')
        plural_dict = {k:v for k,v in plural_dict.items() if v}
        index_set = set([t[0] for t in plural_dict.keys()]) |\
                set(itertools.chain.from_iterable([[t[0] for t in v] for v in plural_dict.values()]))
        entries = df[df['INDEX'].isin(index_set)].to_dict(orient='records')
        entry_dict = {d['INDEX']:lexicon_utils.FormEntry.from_dict(d) for d in entries}

        print('Processing...')
        change_list : list[FormChangeRecord] = []
        for (sing_index, _), plur_list in plural_dict.items():
            for plur_index, _ in plur_list:
                change_rec = compare_entries(entry_dict[sing_index], entry_dict[plur_index])
                change_rec.input_index, change_rec.output_index = sing_index, plur_index
                change_list.append(change_rec)

        change_list_bak = [FormChangeRecord(**dataclasses.asdict(seq)) for seq in change_list]

        print('Filtering')
        for rec in change_list:
            if len(rec.change_sequences) == 1:
                continue
            ending_lens = [len(get_new_ending(seq)) for seq in rec.change_sequences]
            max_len = max(ending_lens)
            rec.change_sequences = [seq for seq, seq_len in zip(rec.change_sequences, ending_lens)
                                                         if seq_len==max_len]

# {("'ă", "'ăi"), ('u', 'ui'), ("'oa", "'oi"), ('l', ''), ("'o", "'oi"), ("'a", "'ei"),
# ("'i", "'ii"), ("'ă", 'ăi'), ('i', 'i'), ('i', "'ii"), ("'ă", 'e'), ("'u", "'ui"),
# ("'â", "'âi"), ('i', 'e'), ("'a", "'ai"), ("'ea", "'ei"), ("'i", 'ii'), ('e', 'e'),
# ("'a", "'ăi"), ('i', 'ii'), ("'e", "'ei")}

    def fix_noendings(chg_rec : dict):
        # We will chop the last letter from the second member of the tuple. That will be the
        # new ending. Exceptions are ('l', ''), which we leave, and ("'ă", 'e'), which is a mistake
        # to be fixed: desinence=ă, new_ending=e, pop last tuple. ('e', 'e') is a mistake that is 
        # not worth fixing. ('i', 'i') is a screwup of the type troliu/troliuri, where the chg_seq
        # is ('^', '^'), ('t', 't'), ('r', 'r'), ("'o", 'o'), ('l', 'l'),
        # ('', "'i"), ('', 'u'), ('', 'r'), ('i', 'i')]
        # 115294/115310 should be elim (emanatie/emanari)
        if chg_rec['new_ending']: # only change when no endings
            return
        final : tuple[str, str] = chg_rec['root_chg'][-1]
        if final == ('l', ''): # leave as is
            return
        if final == ('e', 'e'):
            chg_rec['eliminate'] = True
            return
        if final == ("'ă", 'e'):
            chg_rec.update({'desinence':'ă', 'new_ending':'e'})
            chg_rec['root_chg'].pop()
            return
        if final == ('i', 'i'):
            chg_rec['eliminate'] = True
            return
        chg_rec['new_ending'] = final[1][-1]
        chg_rec['root_chg'][-1] = (final[0], final[1][:-1])
        
