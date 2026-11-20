import pandas as pd
import re
from generate_data import Change, ChangeSequence, ChangeSequenceFromString, change_seq_to_words

REGULAR_CONSONANT_MUTATIONS = [
    {'c', 'c'},
    {'s', 's'},
    {'d', 'z'},
    {'t', 't'}
]

def get_different_finale(chg_seq: ChangeSequence) -> ChangeSequence:
     i = len(chg_seq)-1
     while i >= 0:
         if chg_seq[i].i == chg_seq[i].o:
             return chg_seq[i+1:]
         i -= 1
     return chg_seq

def get_desinences(chg_seq: ChangeSequence) -> tuple[str, str, int]:
    different = get_different_finale(chg_seq)
    des_in = [chg.i for chg in different if chg.i not in {'.', '0'}]
    des_out = [chg.o for chg in different if chg.o not in {'.', '0'}]
    return ''.join(des_in), ''.join(des_out), len(different)


df = pd.read_csv('./lang_data/change_records.doom.v2.work4.tsv', sep='\t')

change_sequences = [
    ChangeSequenceFromString(s) for s in df['phon_changes']
]

