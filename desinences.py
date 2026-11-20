from collections import defaultdict

import pandas as pd
import re
from generate_data import Change, ChangeSequence, ChangeSequenceFromString, change_seq_to_words, ChangeSequenceToString

REGULAR_CONSONANT_MUTATIONS = [
    {'c', 't'}, # caleasca / calesti
    {'s', 'ș'},
    {'d', 'z'},
    {'t', 'ț'},
    {'ci̯', 'c'}, # serviciu - servicii
    {'gi̯', 'g'}, # elogiu - elogii
    {'z', 'j'},
    {'d', 'j'},

]

def get_different_finale(chg_seq: ChangeSequence) -> ChangeSequence:
     i = len(chg_seq)-1
     while i >= 0:
         if chg_seq[i].i == chg_seq[i].o or chg_seq[i].set() in REGULAR_CONSONANT_MUTATIONS:
             return chg_seq[i+1:]
         i -= 1
     return chg_seq

DESINENCE_SG_FIX = {
    "'ă u̯" : -1, # des is u
    "e̯'a": 0,
    'l': 0,
    "'a l e": -1, # des is e
    "ghe̯ 'a": 0,
    "e̯'a u̯": -1, # des is u
    "ce̯ 'a": 0,
    "che̯ 'a": 0,
    "l e": -1, # des is e - piele
    "'a": 0,
    "ge̯ 'a": 0,
    "'î u̯": -1 # des is u
}

BLANKS = {'.', '0', ''}
SG_DES = {'ă', 'e', 'u', 'u̯'}

def get_desinences(seq: ChangeSequence) -> tuple[ChangeSequence, ChangeSequence]:
    # move up to where singular ends
    i = len(seq)-1
    if seq[i].i == seq[i].o == 'ĭ': # avoid screwup
        i -= 1
    while i > 0:
        if seq[i].i not in BLANKS:
            break
        i -= 1
    # does the sg end in SG_DES?
    if seq[i].i in SG_DES or seq[i].tuple() == ('ĭ', 'e'):
        i -= 1
        while i > 0:
            if seq[i].i in BLANKS or \
                    (seq[i].set().issubset({'i', 'i̯', 'ĭ'}) and seq[i].tuple != ('i', 'i')):
                 # or both are short i, or long i?
                i -= 1
            else:
                break
    return seq[:i+1], seq[i+1:]        

# ignore i/i change when looking for vowel changes (('u', '.urĭ'), [("pr'a.csi̯", "pr'a.csi"), ("d'o.li̯", "d'o.li")])
# ('e', ''), [("bă.t'a.i̯", "bă.t'ăi̯")

df = pd.read_csv('./lang_data/change_records.doom.v2.work5.tsv', sep='\t').fillna('')

data_list = df.to_dict(orient='records')

for d in data_list:
    if d['error'] not in {'', 'OK', 'OK0', 'OK1'} or not d['phon_changes']:
        continue
    chg_seq = ChangeSequenceFromString(d['phon_changes'])
    radical_seq, desinence_seq = get_desinences(chg_seq)
    radical_words, des_words = change_seq_to_words(radical_seq), change_seq_to_words(desinence_seq)
    des_sg, des_pl = des_words
    if not des_pl:
        i = len(radical_seq)-1
        while i >= 0:
            if radical_seq[i].o not in BLANKS:
                des_pl = f"({radical_seq[i].o})"
                break
            i -= 1
        else:
            print('Error ',  radical_words, des_words)
    d['radical_seq'] = ChangeSequenceToString(radical_seq)
    d['desinence_seq'] = ChangeSequenceToString(desinence_seq)
    d['desinence_chg'] = des_sg + "/" + des_pl


# df = df[df['error'].isin({'', 'OK', 'OK0', 'OK1'})]

# change_sequences = [
#     ChangeSequenceFromString(s) for s in df['phon_changes'] if s
# ]

# word_dict = {change_seq_to_words(seq):seq for seq in change_sequences}

# des_dict = defaultdict(list)

# for chg_seq in change_sequences:
#     radical_seq, desinence_seq = get_desinences(chg_seq)
#     radical_words, des_words = change_seq_to_words(radical_seq), change_seq_to_words(desinence_seq)
#     print('/ '.join(radical_words),
#           '\t', '/ '.join(des_words))
#     des_dict[des_words].append(radical_words)

# des_list = [(v, k) for k,v in des_dict.items()]
# des_list.sort(key=lambda t: len(t[0]))