import dataclasses

import pandas as pd

from syll_to_phon import VOWEL_CLUSTERS, VOWELS
import re


@dataclasses.dataclass
class Change:
    i: str
    o: str
    def __str__(self) -> str:
        return f'{self.i if self.i else "0"}/{self.o if self.o else "0"}'
    def __repr__(self) -> str:
        return str(self)
    def set(self) -> set:
        return {self.i, self.o}
    def tuple(self) -> tuple:
        return self.i, self.o

ChangeSequence = list[Change]

def ChangeSequenceFromString(seq_str: str) -> ChangeSequence:
    return [Change(*chg.split('/')) for chg in seq_str.split(' ')]

def ChangeSequenceToString(chg_seq: ChangeSequence) -> str:
    return ' '.join([str(chg) for chg in chg_seq])

def change_seq_to_words(chg_seq: ChangeSequence) -> tuple[str, str]:
    w1 = [chg.i for chg in chg_seq if chg.i != '0']
    w2 = [chg.o for chg in chg_seq if chg.o != '0']
    return ''.join(w1), ''.join(w2)

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

def get_surrounding_sounds(chg_seq: ChangeSequence, index: int) -> dict[str, str]:
    # returns before_in, before_out, after_in, after_out
    params = [  ('sound_before_in', False, True),
                ('sound_before_out', False, False),
                ('sound_after_in', True, True),
                ('sound_after_out', True, False)
    ]
    return {t[0]: get_next_sound(chg_seq, index, forwardFlag=t[1], inFlag=t[2])
                        for t in params}


DESINENCE_FAMILIES = {
    'ă/e':['ă/e', ],
    'ă/i':['ă/(cĭ)', 'ă/(gĭ)', 'ă/ĭ'],
    'ă/uri':['ă/urĭ',],
    'e/i':['.i̯e/i̯', 'e/(cĭ)', 'e/(gĭ)', 'e/i̯', 'e/ĭ', 'i.e/ii̯', '.e/i̯', 'e/(chĭ)', ],
    'e/uri':['e/urĭ', ],
    '0/e':['/e', ],
    '0/i':['/i̯', '/ĭ', ],
    '0/le':['/.le', ],
    '0/uri':['/.urĭ', '/urĭ', ],
    'u/i':['i̯u/ii̯', 'u/i', 'u/ii̯', 'u̯/i̯', ],
    'u/e':['u/e', 'u̯/.i̯e', 'u̯/.e', ],
    'u/uri':['i̯u/i.urĭ', 'u/urĭ', 'u̯/.urĭ', ],
    'irregular':['/e.te', '/enĭ', 'u̯/.le', 'u̯/.u̯ă', ], # capete, oameni, rele, oua, 
}

DES_TO_FAMILY = {}
for k, vl in DESINENCE_FAMILIES.items():
    for v in vl:
        DES_TO_FAMILY[v] = k

if __name__ == "__main__":
    FILENAME = './lang_data/change_records.doom.v2.work5.tsv'

    df = pd.read_csv(FILENAME, sep='\t').fillna('')
    df = df[df['error'].isin({'', 'OK', 'OK1', 'OK0'})]
    df = df[df['desinence_seq'] != '']

    df_list = df.to_dict(orient='records')

    TARGET_INDEX = -1


    data_list = []
    for d0 in df_list:
        radical_seq = ChangeSequenceFromString(d0['radical_seq'])
        des_seq = ChangeSequenceFromString(d0['desinence_seq'])
        des_chg = d0['desinence_chg']
        vw_chgs = [chg for chg in radical_seq if string_has_full_vowel(chg.i)
                                            or string_has_full_vowel(chg.o)]
        words = change_seq_to_words(radical_seq)
        if not vw_chgs:
            print('Error! no vowels in word radicals ', words)
            continue
        target_index = TARGET_INDEX if TARGET_INDEX >= 0 else len(vw_chgs) + TARGET_INDEX
        target_chg = vw_chgs[target_index]
        chg_before = vw_chgs[target_index-1] if target_index-1 >= 0 else None
        # chg_after =
        sound_chg_index = (radical_seq + des_seq).index(target_chg)
        d = {
                'word':words,
                'vw0_in':target_chg.i if vw_chgs else '',
                'vw0_out':target_chg.o if vw_chgs else '',
                'des_change':des_chg,
                'des_change_family': DES_TO_FAMILY[des_chg] if des_chg in DES_TO_FAMILY else 'irregular',
                'num_vw_radical':len(vw_chgs),
                'sound_before_des': get_next_sound(radical_seq, len(radical_seq), False, True),
                'vw_before_in': chg_before.i if chg_before else '',
                'vw_before_out': chg_before.o if chg_before else '',
            } | get_surrounding_sounds(radical_seq + des_seq, sound_chg_index)
        d = {k:str(v) for k,v in d.items()}
        data_list.append(d)

    df_save = pd.DataFrame(data_list)

    df_save.to_csv('./train_data/chg.doom2.work5.tsv', sep='\t')        
        
