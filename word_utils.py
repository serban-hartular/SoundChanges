import pandas as pd

def specify_lemma(r : dict) -> dict:
    if r['lemma'] == '=':
        r['lemma'] = r['form']
    return r

SYLL_SEP = '.'
STRESS_MARK = "'"

def get_stressed_syllable(syll_str : str, stress_str : str) -> int:
    sylls = syll_str.split(SYLL_SEP)
    if len(sylls) == 1:
        if STRESS_MARK not in stress_str:
            return 0
        else:
            raise Exception('stress mark in single syllable')
    if STRESS_MARK not in stress_str:
        raise Exception('stress mark missing')
    if stress_str.count(STRESS_MARK) > 1:
        raise Exception('multiple stress marks')
    str_index = 0
    stressed_syllable_index = -1
    for syll_count, syllable in enumerate(sylls):
        for ch1 in syllable:
            if stress_str[str_index] == STRESS_MARK:
                stressed_syllable_index = syll_count
                str_index += 1
            ch2 = stress_str[str_index]
            if ch1.lower() != ch2.lower():
                raise Exception(f'mismatch: {ch1} vs {ch2}')
            str_index += 1
    return stressed_syllable_index

VOWELS = {'a', 'e', 'i', 'o', 'u', '@', '1'}
SEMIVOWELS = {'j', 'w', 'o_X', 'e_X'}
I_FINAL = 'i_0'
CONSONANTS = {'ts', 'r', 'g_j', 'f', 'c', 'v', 'k', 'Z', 'S', 'k_j', 'tS', 'm',
              'gZ', 'd', 's', 'n', 'b', 'l', 'z', 'p', 'g', 't', 'h'}

ALL_PHONS = VOWELS | CONSONANTS | SEMIVOWELS | {I_FINAL}

def count_syllables(syll_str : str, phon_str : str) -> int:
    count1 = syll_str.count(SYLL_SEP) + 1
    phon_list = phon_str.split(' ')
    count2 = len([ch for ch in phon_list if ch in VOWELS])
    if count1 != count2:
        raise Exception('syll count mismatch')
    return count1

def stress_and_sylls(r : dict) -> dict:
    try:
        count = count_syllables(r['syll'], r['phon'])
        r['syll_count'] = count
        r['comment'] = ''
    except Exception as e:
        r['syll_count'] = -1
        r['comment'] = str(e)
    try:
        stressed_syl = get_stressed_syllable(r['syll'], r['stress'])
        r['stress_syl'] = stressed_syl
    except Exception as e:
        r['stress_syl'] = 0 if str(e) == 'stress mark in single syllable' else -1
        r['comment'] = r['comment'] + ('|' if r['comment'] else '') + str(e)

    return r

def lemma_get_root(phon : list[str], xpos : str) -> list[str]:
    if xpos[0] == 'V': # verb
        if phon[-2:] == ['e_X', 'a']:
            return phon[:-2]
        elif phon[-1] in {'a', 'e', 'i', '1'}:
            return phon[:-1]
        else:
            raise Exception(f'Unkown infinitive {phon}')
    to_drop = set(VOWELS + SEMIVOWELS + [I_FINAL])
    if phon[-1] in to_drop:
        return phon[:-1]
    return phon


if __name__ == "__main__":
    pass

