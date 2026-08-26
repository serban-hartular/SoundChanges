import itertools

from ortho_form_comparison2 import FormChangeRecord
from lexicon_utils import ORTHO_VOWELS

import pandas as pd
import re

from lexicon_utils import SYLLABLE_MARK, STRESS_MARK

def syllable_form_to_chunks(syll_form : str) -> list[str]:
    """Split syllable form into chunks of vowels/consonants, while respecting hiatuses.
    e.g.: a.u.re'an -> a, u, r, ea, n, but praz.nic -> pr a zn i c"""
    vowels_regex = f"[{''.join(ORTHO_VOWELS)}{STRESS_MARK}]+" # add stress mark
    consonants_regex = f"[^{''.join(ORTHO_VOWELS)}{STRESS_MARK}]+"
    syll_form = re.sub(fr'({consonants_regex})\.({consonants_regex})', r'\1\2', syll_form)
    sylls = syll_form.split('.')
    chunks = []
    for syll in sylls:
        split = list(re.finditer(f'{vowels_regex}|{consonants_regex}', syll))
        chunks.extend([m.group() for m in split])
    return chunks

def get_change_context(chg_rec : FormChangeRecord, chg_index : int) ->\
                                                dict[int, dict]:
    chg_seq = chg_rec.change_sequences[0]
    # chg_index unchanged because length lists below start with a 0.

    contexts = {}

    # input and output in chunks
    chunks = [syllable_form_to_chunks(s) for s in (chg_rec.input_syll, chg_rec.output_syll)]
    for in_out in (0, 1): # input vs output
        # span of change in characters for input and output
        change_start = sum([len(chg[in_out]) for chg in chg_seq[1:chg_index]]) # skip initial ^
        change_end = change_start + len(chg_seq[chg_index][in_out])
        chunk_len, change_chunk = 0, -1
        for chunk_index, chunk in enumerate(chunks[in_out]):
            chunk_len += len(chunk)
            if chunk_len > change_start: # this is the chunk
                change_chunk = chunk_index
                chunk_len -= len(chunk)
                break
        else: # not found
            raise Exception('Change chunk not found')
        change_start -= chunk_len # change position within chunk
        change_end -= chunk_len
        immediate_back, immediate_forward = chunk[:change_start], chunk[change_end:]
        context = {'back':{'immediate':immediate_back}, 'forward':{'immediate':immediate_forward}}
        mod_chunks = ['^', '^'] + chunks[in_out] + ['$', '$']
        change_chunk += 2 # because we added 2 items at the beginning
        for delta in (-1, 1): # backwards and forwards
            consonants = mod_chunks[change_chunk + delta]
            vowels = mod_chunks[change_chunk + 2*delta]
            if consonants[0] in (ORTHO_VOWELS|{"'"}): # we have no consonants
                vowels = consonants
                consonants = ''
            context['back' if delta == -1 else 'forward'].update({'consonants':consonants, 'vowels':vowels})
        contexts['in' if in_out == 0 else 'out'] = context
    return contexts

def flatten_context(contexts : dict) -> dict[str, str]:
    items = {}
    for in_out, v0 in contexts.items():
        for forward_back, v1 in v0.items():
            for position, value in v1.items():
                items['_'.join([position, forward_back, in_out])] = value
    return items       

def find_changes_from(change_from : str, change_sequence : list[tuple[str, str]]) -> list[int]:
    index_list = []
    for i, chg in enumerate(change_sequence):
        if chg[0] == change_from:
            index_list.append(i)
    return index_list

if __name__ == "__main__":
    df = pd.read_csv('./train_data/change_records.tsv', sep='\t')
    df = df.fillna('')
    change_records = [FormChangeRecord.from_dict(r) for r in df.to_dict(orient='records')]

    data_list = []

    for change_from in {"'a", "e'a", 'i', 'e', "'i", "o'a", "'o", "'u", "'â",
                        'ea', 'a', "'ă", 'o', "'e",  'ă'}: #'u', 'â',
        filtered_changes = [(chg_rec, find_changes_from(change_from, chg_rec.change_sequences[0]))
                        for chg_rec in change_records]
        filtered_changes = [(chg_rec, chg_indices) for chg_rec, chg_indices in filtered_changes if chg_indices]
        target_changes = list(itertools.chain.from_iterable(
            [[(chg_rec, i) for i in chg_indices] for chg_rec, chg_indices in filtered_changes]))
        for chg_rec, chg_index in target_changes:
            change_sequence = chg_rec.change_sequences[0]
            d = {'form':'/'.join([chg_rec.input_form, chg_rec.output_form]),
                'ph_initial':change_sequence[chg_index][0],
                'ph_changed':change_sequence[chg_index][1],
                 'chg_index':chg_index,
            }
            d.update(flatten_context(get_change_context(chg_rec, chg_index)))
            data_list.append(d)

    df = pd.DataFrame(data_list)
    df.to_csv('./train_data/change_data_all.tsv', sep='\t')
        
# Counter({"'a": 1926, "'o": 362, "e'a": 328, "o'a": 79, "'i": 50, 'o': 31, "'â": 28,
# 'ă': 27, "'ă": 24, 'i': 23, 'a': 21, "'e": 21, 'e': 21, 'u': 16, "'u": 6, 'â': 4, 'ea': 2})

