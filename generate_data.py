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

def get_vowel_changes(chg_seq: ChangeSequence) -> tuple[list[Change], Change|None]:
    # return vowel changes plus optional final non-vowel change (as in pom/pomi)
    vw_chgs = [chg for chg in chg_seq if string_has_full_vowel(chg.i)
                                     or string_has_full_vowel(chg.o)]
    tail = None
    if not vw_chgs:
        print('No vowels in ', chg_seq)
        return [], None
    if ((chg_seq[-1].i == 'ĭ' or chg_seq[-1].o == 'ĭ') and vw_chgs[-1] is not chg_seq[-1])\
            or vw_chgs[-1].i == vw_chgs[-1].o: # something must have changed, no?
        tail = chg_seq[-1]
    return vw_chgs, tail

def get_next_sound(chg_seq: ChangeSequence, index: int, forwardFlag: bool, inFlag:bool) -> str:
    delta = 1 if forwardFlag else -1
    while True:
        index += delta
        if index < 0 or index >= len(chg_seq):
            return ''
        chg = chg_seq[index]
        sound = chg.i if inFlag else chg.o
        if sound not in {'.', '0'}:
            return sound


if __name__ == "__main__":

    data_list = []
    for chg_seq in change_seqs:
        key = str(change_seq_to_words(chg_seq))
        vowel_changes, tail = get_vowel_changes(chg_seq)
        if tail is not None:
            vowel_changes.append(tail)
        for i, vw_chg in enumerate(vowel_changes[:-1]):
            vw_in, vw_out = vw_chg.i, vw_chg.o
            vw_chg_next = vowel_changes[i+1] if i+1<len(vowel_changes) else ''
            vw_chg_before = vowel_changes[i-1] if i-1>=0 else ''
            chg_seq_index = chg_seq.index(vw_chg)
            chg_before = chg_seq[chg_seq_index-1] if chg_seq_index-1>=0 else Change('', '')
            chg_after = chg_seq[chg_seq_index+1] if chg_seq_index+1<len(chg_seq) else Change('', '')
            sound_in_before = get_next_sound(chg_seq, chg_seq_index, forwardFlag=False, inFlag=True)
            sound_out_before = get_next_sound(chg_seq, chg_seq_index, forwardFlag=False, inFlag=False)
            sound_in_after = get_next_sound(chg_seq, chg_seq_index, forwardFlag=True, inFlag=True)
            sound_out_after = get_next_sound(chg_seq, chg_seq_index, forwardFlag=True, inFlag=False)

            syll_count_before, syll_count_after = key[0].count('.')+1, key[1].count('.')+1
            d = {
                'ID':key,
                'vw_in':vw_in, 'vw_out':vw_out,
                'vw_chg_before':vw_chg_before, 'vw_chg_next':vw_chg_next,
                'chg_before':chg_before, 'chg_after':chg_after,
                'sound_in_before':sound_in_before, 'sound_in_after':sound_in_after,
                'sound_out_before':sound_out_before, 'sound_out_after':sound_out_after,
                'syll_count_before':syll_count_before, 'syll_count_after':syll_count_after,
            }
            d = {k:str(v) for k,v in d.items()}
            data_list.append(d)

    df_save = pd.DataFrame(data_list)

    df_save.to_csv('./train_data/chg.doom2.work3.tsv', sep='\t')        
        
