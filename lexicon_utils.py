import dataclasses
import re

from word_utils import VOWELS, SEMIVOWELS, I_FINAL, CONSONANTS

VOCALIC = VOWELS | SEMIVOWELS

ORTHO_VOWELS = set('aeiouăîâ')

STRESS_MARK = "'"
SYLLABLE_MARK = '.'

@dataclasses.dataclass
class FormEntry:
    form : str
    lemma : str
    xpos : str
    syllables : str
    stress : str
    phon : str

    @staticmethod
    def from_dict(d : dict) -> 'FormEntry':
        return FormEntry(**{k:v for k,v in d.items() if k in FormEntry.__annotations__})



class FormEntrySanityChecks:
    @staticmethod
    def count_syllable_centers(e : FormEntry) -> int:
        return len([s for s in e.phon.split() if s in VOWELS])

    @staticmethod
    def stress_check(e : FormEntry) -> str:
        if e.form.lower() != e.stress.replace(STRESS_MARK, '').lower():
            return 'Stress form mismatch'
        if STRESS_MARK in e.stress:
            i = e.stress.find(STRESS_MARK)
            if i > len(e.stress)-2 or e.stress[i+1] not in ORTHO_VOWELS:
                return 'Stress mark not before vowel'
        elif FormEntrySanityChecks.count_syllable_centers(e) != 1:
            return 'No stress mark but multiple syllables'
        return ''
    @staticmethod
    def syll_check(e : FormEntry) -> str:
        if e.form.lower() != e.syllables.replace(SYLLABLE_MARK, '').lower():
            return 'Syllable form mismatch'
        sylls = e.syllables.split(SYLLABLE_MARK)
        for syll in sylls:
            if not syll:
                return 'Empty syllable'
            if not ORTHO_VOWELS.intersection(syll):
                return 'Syllable has no vowels'
            if syll == sylls[-1] and syll[-1] == 'i':
                syll = syll[:-1] # remove final i in case it is i_0
            # count number of syllable clusters -- there should only be one
            if len(re.findall(f'[{''.join(ORTHO_VOWELS)}]+', syll)) > 1:
                return 'Too many vowel clusters'
        return ''

    
#RETEROM_COLUMNS = ['form', 'lemma', 'xpos', 'syllables', 'stress', 'phon']
RETEROM_COLUMNS = list(FormEntry.__annotations__)



def err_msg(e : FormEntry) -> str:
    fn_list = [FormEntrySanityChecks.stress_check, FormEntrySanityChecks.syll_check]
    msgs = [fn(e) for fn in fn_list]
    msgs = [s for s in msgs if s]
    return ''.join(msgs)


def stress2syllables(e : FormEntry, stress_sym : str = '', syll_sym : str = SYLLABLE_MARK) -> str:
    stress_sym, syll_sym = stress_sym or STRESS_MARK, syll_sym or SYLLABLE_MARK
    i_stress, i_syll, out_str = 0, 0, ''
    while i_stress < len(e.stress) and i_syll < len(e.syllables):
        c_stress, c_syll = e.stress[i_stress], e.syllables[i_syll]
        if c_stress == c_syll: # same symbol
            out_str += c_stress
            i_stress, i_syll = i_stress + 1, i_syll + 1
        elif c_syll == syll_sym: # syllable edge symbol
            out_str += c_syll
            i_syll += 1
        elif c_stress == stress_sym: # stress symbol
            out_str += c_stress
            i_stress += 1
        else:
            raise Exception(f'Error merging {e.stress}, {e.syllables} at chars {c_stress}, {c_syll}')
    return out_str


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv('./lexicon/lexicon_reterom.v1.mod.tsv', sep='\t')
    #df['include'] = df.apply(lambda row: filter_fn(row['xpos']), axis=1)
    #df = df[df['include']==1]
    


