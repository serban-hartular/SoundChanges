from word_utils import CONSONANTS, VOWELS, SEMIVOWELS, I_FINAL, ALL_PHONS
from word_distance import Transition, ChangeSequence, WordTransformation
from sound_change_eval import get_all_change_sequences, score_fn
import pandas as pd
from collections import Counter
import dataclasses
import pickle

KEY_COLUMN = 'key'

def load_tsv(filepath : str) -> pd.DataFrame:
    return pd.read_csv(filepath, sep='\t', encoding='utf-8')

def save_tsv(filepath : str, df : pd.DataFrame):
    df.to_csv(filepath, sep='\t', encoding='utf-8')

def get_by_key(df : pd.DataFrame, key: str, params: list[str], unique = True) -> list|list[list]:
    rows = df[df[KEY_COLUMN]==key].to_dict(orient='records')
    if unique and len(rows) != 1:
        raise Exception(f'Key {key} not {"found" if len(rows)==0 else "unique"}!')
    data = [[d[k] for k in params] for d in rows]
    return data[0] if unique else data

def get_unique(df : pd.DataFrame, col_name : str) -> set[tuple[str, str]]:
    unique = {l for l, c in Counter(df[col_name]).items() if c==1}
    df_unique = df[df[col_name].isin(unique)]
    return set(zip(df[col_name], df[KEY_COLUMN]))

@dataclasses.dataclass
class FormRecord:
    key: str
    phon: str
    def to_json(self):
        return dataclasses.asdict(self)
    @staticmethod
    def from_json(d : dict) -> 'FormRecord':
        return FormRecord(**d)

@dataclasses.dataclass
class ChangeRecord:
    form1 : FormRecord
    form2 : FormRecord
    change_seq : list[tuple[str, str]]|None = None
    def to_json(self):
        return {'form1':self.form1.to_json(), 'form2':self.form2.to_json(), 'change_seq':self.change_seq}
    @staticmethod
    def from_json(d : dict) -> 'ChangeRecord':
        cr = ChangeRecord(**d)
        cr.form1 = FormRecord.from_json(cr.form1)
        cr.form2 = FormRecord.from_json(cr.form2)
        return cr

def allowed(s1 : str, s2 : str) -> bool:
        if not s1 or not s2:
            return True
        if (s1 in VOWELS and s2 in CONSONANTS) or (s2 in VOWELS and s1 in CONSONANTS):
            return False
        return True

def cleanup(phon : str) -> str:
    phon = phon.replace('.', '')
    phon_list = phon.split()
    new_list = []
    for ph in phon_list:
        if ph not in ALL_PHONS:
            ph_list = list(ph)
            if not ALL_PHONS.issuperset(ph_list):
                raise Exception(f'Unknown symbol(s) {set(ph_list).difference(ALL_PHONS)}')
            new_list.extend(ph_list)
        else:
            new_list.append(ph)
    return ' '.join(new_list)


def get_chg_rec_data(cr : ChangeRecord, chg_of_interest : tuple[str, str]) -> list[str]:
    data = list(chg_of_interest)
    suffix = cr.change_seq[-2][1] # type: ignore
    orig_desinence = cr.form1.key.split('|')[0][-1]
    if orig_desinence not in 'aeiouăîâ':
        orig_desinence = ''
    stress_change = cr.form1.key.split('|')[-1] != cr.form2.key.split('|')[-1]
    # get surroundings
    phon = cr.form1.phon.split()
    i = 0
    for chg in cr.change_seq:
        if chg == chg_of_interest:
            break
        if chg[0] == phon[i]:
            i += 1
    before, after = phon[i-1] if i > 0 else '', phon[i+1] if i < len(phon)-1 else ''
        
    data += [before, after, cr.form1.key, cr.form2.key, orig_desinence, suffix, str(stress_change)]
    return data


if __name__ == "__main__":
    df_plurs = load_tsv('./lexicon/df_nplur.tsv')
    df_roots = load_tsv('./lexicon/flex_lemmas6.tsv')

    root_lemmas = dict(get_unique(df_roots, 'form'))
    plur_lemmas = dict(get_unique(df_plurs, 'lemma'))

    lemmas = list(set(root_lemmas).intersection(plur_lemmas))
    lemmas.sort()

    change_list = []

    for i, lemma in enumerate(lemmas):
        if i%100 == 0:
            print(f'{i} of {len(lemmas)}')
        root_key = root_lemmas[lemma]
        root_rec = FormRecord(key=root_key, phon=cleanup(get_by_key(df_roots, root_key, ['root'])[0]))
        for row in df_plurs[df_plurs['lemma']==lemma].to_dict(orient='records'):
            plur_rec = FormRecord(key=row[KEY_COLUMN], phon=cleanup(row['phon']))
            cr = ChangeRecord(root_rec, plur_rec)
            wt = WordTransformation(cr.form1.phon.split(), cr.form2.phon.split(), allowed_fn=allowed, cost_fn=score_fn)
            wt.compute_change_sequences()
            all_seqs = get_all_change_sequences(wt.change_sequences)
            cr.change_seq = [(t.d_in, t.d_out) for t in all_seqs[0][0]]
            change_list.append(cr)
        
    with open('./nplur_changes.p', 'wb') as handle:
        pickle.dump([cr.to_json() for cr in change_list], handle)
    