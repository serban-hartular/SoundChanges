import dataclasses

import lemmatize
import lexicon_utils as lu
import pandas as pd
import re

import word_distance

def chop_off_letters(s: str, count: int) -> str:
    while s and count > 0:
        if s[-1].isalpha():
            count -= 1
        s = s[:-1]
    return s

TO_GROUP = ["'"+vw for vw in lu.ORTHO_VOWELS] + ["oa", "ea", "o'a", "e'a"]

def group_letters(word: str, to_group: list[str]) -> list[str]:
    to_group.sort(key=lambda s: -len(s)) # sort long to short
    regex = '|'.join(to_group + [r"\w"])
    return [m.group() for m in re.finditer(regex, word)]

DESCENDING2 = ['ai', 'au', 'ău', 'ăi','ei', 'eu', 'ii', 'iu', 'îi', 'âi', 'îu', 'âu', 'ou', 'oi',
               'ui', 'ue']
ASCENDING2 = ['ea', 'eo', 'ia', 'ie', 'iu', 'oa', 'ua', 'uă', 'uî', 'uâ', 'io']
ASCENDING3 = ['eoa', 'ioa', ]
CENTERED3 = ['eai', 'eau', 'iai', 'iau', 'iei', 'ieu', 'ioi', 'oai', 'uai', 'uau']

def stress_mark_cleanup(syll_form : str) -> str:
    """Find first vowel cluster, insert stress mark"""
    m = re.search(f"[{''.join(lu.ORTHO_VOWELS)}]+", syll_form)
    if m is None:
        raise Exception('Word has no vowels: ' + syll_form)
    span = m.span()
    cluster = m.group()
    if len(cluster)==1:
        cluster = "'" + cluster
    else:
        if cluster in DESCENDING2:
            cluster = "'" + cluster
        elif cluster in ASCENDING2:
            cluster = cluster[0] + "'" + cluster[1]
        elif cluster in ASCENDING3:
            cluster = cluster[:2] + "'" + cluster[2]
        elif cluster in CENTERED3:
            cluster = cluster[0] + "'" + cluster[1:]
        else:
            raise Exception('Unknown di/triphthong ' + cluster + " in " + syll_form)
    return syll_form[:span[0]] + cluster + syll_form[span[1]:]





def process_pair(e_sg : lu.FormEntry, e_pl : lu.FormEntry,
                 nfe_list : list[lemmatize.NounFlexionEntry]) -> dict:
    nfe = lemmatize.filter_nfes(e_sg.form.replace("â", "î"), e_pl.form.replace("â", "î"), nfe_list)
    if nfe is None:
        raise Exception(f"Could not find nfe for {e_sg.form} / {e_pl.form}")
    desinences = nfe.desin_sg, nfe.desin_pl
    syll_forms = lu.stress2syllables(e_sg), lu.stress2syllables(e_pl)
    syll_forms = tuple(stress_mark_cleanup(s) if "'" not in s else s for s in syll_forms)
    syll_forms_chopped = tuple(chop_off_letters(s, len(d)) for s,d in zip(syll_forms, desinences))
    chunks = tuple(group_letters(word, TO_GROUP) for word in syll_forms_chopped)

    return {
        'input_form': e_sg.form,
        'input_syll': syll_forms[0],
        'input_des': desinences[0],
        'input_chunks':chunks[0],
        'output_form': e_pl.form,
        'output_syll': syll_forms[1],
        'output_des': desinences[1],
        'output_chunks':chunks[1],
    }

def change2str(chg : tuple[str, str]) -> str:
    return f'{chg[0] or '0'}/{chg[1] or '0'}'

def change_from_str(s: str) -> tuple[str, str]:
    t = tuple(t.strip() for t in s.split('/')) 
    t = tuple('' if s == '0' else s for s in t)
    return t # pyright: ignore[reportReturnType]

def chg_seq_2_str(seq: list[tuple[str, str]]) -> str:
    return ' '.join([change2str(chg) for chg in seq])

def chg_seq_from_str(seq_str : str) -> list[tuple[str, str]]:
    return [change_from_str(s) for s in seq_str.split()]

@dataclasses.dataclass
class FormChangeRecord:
    input_form : str
    input_syll : str
    input_des : str
    output_form : str
    output_syll : str
    output_des : str
    input_index : int
    output_index : int
    change_sequences : list[list[tuple[str, str]]]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d['change_sequences'] = ' | '.join([chg_seq_2_str(seq) for seq in self.change_sequences])
        return d

    @staticmethod
    def from_dict(d : dict) -> 'FormChangeRecord':
        rec = FormChangeRecord(**{k:v for k,v in d.items() if k in FormChangeRecord.__annotations__})
        rec.change_sequences = [chg_seq_from_str(s) for s in d['change_sequences'].split('|')]
        return rec

VOWEL_CLUSTER_SET = lu.ORTHO_VOWELS | {"'"}

def allowed_fn(in_str : str, out_str : str) -> bool:
    """Dissalow exchanging vowels for consonants or vice_versa"""
    if not in_str or not out_str: # allow insertions and deletions
        return True
    is_vowel_cluster = [bool(VOWEL_CLUSTER_SET.intersection(s)) for s in (in_str, out_str)]
    return is_vowel_cluster[0] == is_vowel_cluster[1]

def generate_change_record(e_in : lu.FormEntry, e_out : lu.FormEntry,
                           nfe_list : list[lemmatize.NounFlexionEntry],
                           index_in = -1, index_out = -1) -> FormChangeRecord:
    pair_info = process_pair(e_in, e_out, nfe_list)
    input_chunks, output_chunks = pair_info.pop('input_chunks'), pair_info.pop('output_chunks')
    wt = word_distance.WordTransformation(input_chunks, output_chunks, allowed_fn=allowed_fn)
    wt.compute_change_sequences()
    if not wt.change_sequences:
        raise Exception('Could not compute change sequences')
    pair_info['change_sequences'] = [[(chg.d_in, chg.d_out) for chg in seq] for seq in
                                              wt.change_sequences]
    pair_info |= {'input_index':index_in, 'output_index':index_out}
    return FormChangeRecord(**pair_info)
    

if __name__ == "__main__":
    import pickle
    import itertools

    print('Loading')

    # load plural pairs
    with open('./lexicon/plurals_indices.p', 'rb') as handle:
        plural_dict : dict[tuple, list] = pickle.load(handle)
    plural_dict = {k:v for k,v in plural_dict.items() if v}
    plural_pairs = list(itertools.chain.from_iterable([[(k[0], pl[0]) for pl in v] 
                                                           for k,v in plural_dict.items()]))
    t = (136571, 137082)
    if t in plural_pairs: # ghint/giuvaer error
        plural_pairs.remove(t)

    # load noun data
    df = pd.read_csv('./lexicon/nouns.v3.tsv', sep='\t')
    index_set = set(itertools.chain.from_iterable(plural_pairs))
    entry_dict = {d['INDEX']:lu.FormEntry.from_dict(d)
                  for d in df[df['INDEX'].isin(index_set)].to_dict(orient='records')}

    # load flexion classes
    df = pd.read_csv('./lang_data/subst_flex_clase.tsv', sep='\t')
    df = df.fillna('')
    nfe_list = lemmatize.NFE_from_df(df)

    change_records = []
    for indices in plural_pairs:
        e1, e2 = [entry_dict[i] for i in indices]
        ch_rec = generate_change_record(e1, e2, nfe_list, *indices)
        d = ch_rec.to_dict()
        d['change_sequences'] = d['change_sequences'].split('|')[0].strip()
        change_records.append(d)
