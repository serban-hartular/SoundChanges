import dataclasses
from typing import Callable
import itertools

from word_distance import WordTransformation, Transition, ChangeSequence

from word_utils import VOWELS, SEMIVOWELS, I_FINAL, CONSONANTS

VOCALIC = VOWELS | SEMIVOWELS

diphthong_changes = [
    (['o_X', 'a'], ['o']), (['e_X', 'a'], ['e']),
]

diphthong_changes.extend([(l2, l1) for l1,l2 in diphthong_changes])

def copy_t(t : Transition):
    return Transition(d_in=t.d_in, d_out=t.d_out)

def merge_changes(chg_seq : ChangeSequence, merge_table : list[tuple[list, list]]) -> ChangeSequence:
    new_seq : ChangeSequence= []
    i = 0
    while i < len(chg_seq):
        for inputs, results in merge_table:
            seg_len = max([len(inputs), len(results)])
            if [t.d_in for t in chg_seq[i:i+seg_len] if t.d_in] ==  inputs and\
               [t.d_out for t in chg_seq[i:i+seg_len] if t.d_out] == results:
                new_seq.append(Transition(d_in=' '.join(inputs), d_out=' '.join(results)))
                i += seg_len
                break
            # # if new_seq:
            # #     print(new_seq[-1].d_in, inputs[0], chg.d_in, inputs[1], new_seq[-1].d_out + chg.d_out, result)
            # if new_seq and new_seq[-1].d_in == inputs[0] and chg.d_in == inputs[1] \
            #                             and new_seq[-1].d_out + chg.d_out == result:
            #     new_seq[-1] = Transition(d_in=' '.join(inputs), d_out=result)
            #     break
        else:
            new_seq.append(copy_t(chg_seq[i]))
            i += 1
    return new_seq

def merge_suffix(chg_seq : ChangeSequence) -> ChangeSequence:
    suffix_end = len(chg_seq)-1
    i = suffix_end-1
    while chg_seq[i].d_in == '':
        i -= 1
    i += 1
    if i<suffix_end-1:
        return chg_seq[:i] + [Transition(d_in='', d_out=' '.join([t.d_out for t in chg_seq[i:suffix_end]])), chg_seq[-1]]
    else:
        return chg_seq
    
change_scores = {
    0.5 : [('s', 'S'), ('z', 'Z'), ('k', 'tS'), ('g', 'gZ'), ('t', 'ts'), ('d', 'z'),
           (I_FINAL, 'i'), ('j', 'i'), ('i', 'j'), ('w', 'u'), ('w', 'u'), ('o_X', 'o'), ('o', 'o_X'), ('e_X', 'e'), ('e', 'e_X')
        ],
    5 : list(itertools.product(VOWELS, CONSONANTS)) + list(itertools.product(CONSONANTS, VOWELS)),
}    

pair_to_score = {pair:score for score, pair_list in change_scores.items() for pair in pair_list}

def score_fn(d_in : str, d_out : str) -> float:
    if d_in == d_out:
        return 0
    if (d_in, d_out) in pair_to_score:
        return pair_to_score[(d_in, d_out)]
    return 1


def score_sequence(chg_seq : ChangeSequence):
    for chg in chg_seq:
        if chg.d_in == chg.d_out:
            chg.cost = 0
        elif (chg.d_in, chg.d_out) in pair_to_score:
            chg.cost = pair_to_score[(chg.d_in, chg.d_out)]
        else:
            chg.cost = 1

def sort_sequences(chg_seqs : list[ChangeSequence]) -> list[tuple[ChangeSequence, float]]:
    sorted_list = []
    for seq in chg_seqs:
        score_sequence(seq)
        sorted_list.append((seq, sum([chg.cost for chg in seq])))
    sorted_list.sort(key=lambda t: t[1])
    return sorted_list

def get_all_change_sequences(seq_list : list[ChangeSequence]) -> list[tuple[ChangeSequence, float]]:
    new_list = [seq for seq in seq_list]
    for seq in new_list:
        seq = merge_changes(seq, diphthong_changes)
        seq = merge_suffix(seq)
        if seq not in new_list:
            new_list.append(seq)
    return sort_sequences(new_list)

def get_change_seqs(word1 : str|list[str], word2 : str|list[str], sep = ' ', allowed_fn = None, score_fn = None) -> list[tuple[ChangeSequence, float]]:
    if isinstance(word1, str):
        word1 = word1.split(sep) if sep else list(word1)
    if isinstance(word2, str):
        word2 = word2.split(sep) if sep else list(word2)

    wt = WordTransformation(word1, word2, allowed_fn, score_fn)
    wt.compute_change_sequences()
    return get_all_change_sequences(wt.change_sequences)


if __name__ == "__main__":
    def allowed(s1 : str, s2 : str) -> bool:
        if not s1 or not s2:
            return True
        if (s1 in VOWELS and s2 in CONSONANTS) or (s2 in VOWELS and s1 in CONSONANTS):
            return False
        return True
    

    # wt = WordTransformation('b e_X a t'.split(), 'b e ts i'.split(), allowed, score_fn)
    # wt.compute_change_sequences()
    # seq_all = get_all_change_sequences(wt.change_sequences)
    word1, word2 = 'b e_X a t', 'b e ts i'
    seq_all = get_change_seqs(word1, word2, ' ', allowed, score_fn)
    print(seq_all)