import dataclasses

import pandas as pd

@dataclasses.dataclass
class Change:
    i: str
    o: str
    def __str__(self) -> str:
        return f'{self.i}/{self.o}'
    def __repr__(self) -> str:
        return str(self)

ChangeSequence = list[Change]

def ChangeSequenceFromString(seq_str: str) -> ChangeSequence:
    return [Change(*chg.split('/')) for chg in seq_str.split(' ')]

def change_seq_to_words(chg_seq: ChangeSequence) -> tuple[str, str]:
    w1 = [chg.i for chg in chg_seq if chg.i != '0']
    w2 = [chg.o for chg in chg_seq if chg.o != '0']
    return ''.join(w1), ''.join(w2)

FILENAME = './lang_data/change_records.doom.v2.work3.tsv'

df = pd.read_csv(FILENAME, sep='\t').fillna('')

change_seqs = [ChangeSequenceFromString(s) for s in df['phon_changes'] if s]

from syll_to_phon import VOWEL_CLUSTERS, VOWELS
import re

def string_has_full_vowel(s: str) -> bool:
    return bool(re.search(rf'[{''.join(VOWELS)}]^{'u̯'[-1]}', s)) or \
        bool(re.search(rf'[{''.join(VOWELS)}]$', s))

def get_vowel_changes(chg_seq: ChangeSequence) -> list[Change]:
    vw_chgs = [chg for chg in chg_seq if string_has_full_vowel(chg.i)
                                     or string_has_full_vowel(chg.o)]
    if not vw_chgs:
        print('Now vowels in ', chg_seq)
        return []
    if (chg_seq[-1].i == 'ĭ' or chg_seq[-1].o == 'ĭ') and vw_chgs[-1] is not chg_seq[-1]:
        vw_chgs.append(chg_seq[-1])
    return vw_chgs

target = "'ă"
data_list = []
for chg_seq in change_seqs:
    key = str(change_seq_to_words(chg_seq))
    found_index = 0
    for i, chg in enumerate(chg_seq):
        if chg.i == target: # we're on
            found_index += 1
            
    
