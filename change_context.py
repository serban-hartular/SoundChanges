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
        immediate_forward, immediate_back = chunk[:change_start], chunk[change_end:]
        context = {-1:{'immediate':immediate_back}, 1:{'immediate':immediate_forward}}
        mod_chunks = ['^', '^'] + chunks[in_out] + ['$', '$']
        change_chunk += 2 # because we added 2 items at the beginning
        for delta in (-1, 1): # backwards and forwards
            consonants = mod_chunks[change_chunk + delta]
            vowels = mod_chunks[change_chunk + 2*delta]
            if consonants[0] in (ORTHO_VOWELS|{"'"}): # we have no consonants
                vowels = consonants
                consonants = ''
            context[delta].update({'consonants':consonants, 'vowels':vowels})
        contexts[in_out] = context
    return context

         

if __name__ == "__main__":
    df = pd.read_csv('./train_data/change_records.tsv', sep='\t')
    change_records = [FormChangeRecord.from_dict(r) for r in df.to_dict(orient='records')]

    chg_rec = change_records[1012]
    get_change_context(chg_rec, 5)


