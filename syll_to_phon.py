import re
from typing import Any

VOWELS = set('aeiouîă')

VOWELS_TO_SEMI = {
    'i':'i̯',
    'u':'u̯',
    'e':'e̯',
    'o':'o̯'
}

VOCALIC = VOWELS | set(VOWELS_TO_SEMI.values())

FINAL_I = 'ĭ'

VOWEL_CLUSTERS = {
    vw : vw for vw in VOWELS
} | {
    'ea': 'e̯ a',
    'eo': 'e̯ o',
    'ia': 'i̯ a',
    'ie': 'i̯ e',
    'io': 'i̯ o',
    'iu': ['i̯ u', 'i u̯'],
    'oa': 'o̯ a',
    'ua': 'u̯ a',
    'uă': 'u̯ ă',
    'uî': 'u̯ î',
    'ai': 'a i̯',
    'au': 'a u̯',
    'ăi': 'ă i̯',
    'ău': 'ă u̯',
    'ei': 'e i̯',
    'eu': 'e u̯',
    'îi': 'î i̯',
    'îu': 'î u̯',
    'ii': 'i i̯',
    #'iu': 'iu̯',
    'oi': 'o i̯',
    'ou': 'o u̯',
    'ui': 'u i̯',
    'eoa': 'e̯ o̯ a',
    'ioa': 'i̯ o̯ a',
    'eai': 'e̯ a i̯',
    'eau': 'e̯ a u̯',
    'iai': 'i̯ a i̯',
    'iau': 'i̯ a u̯',
    'iei': 'i̯ e i̯',
    'ieu': 'i̯ e u̯',
    'ioi': 'i̯ o i̯',
    'iou': 'i̯ o u̯',
    'oai': 'o̯ a i̯',
    'uăi': 'u̯ ă i̯',
    'ioai': 'i̯ o̯ a i̯', # init i will be merged with c/g later
}


def _clusters_sanity_check():
    for ortho, phon_list in VOWEL_CLUSTERS.items():
        ortho = list(ortho)
        phon_list = phon_list if isinstance(phon_list, list) else [phon_list]
        for phon in phon_list:
            id_string = f'"{''.join(ortho)}", "{phon}"'
            phon = phon.split(' ')
            if len(phon) != len(ortho):
                raise Exception('Unequal lenths in ' + id_string)
            vowel_count = 0
            for ort, ph in zip(ortho, phon):
                if not (ph in VOCALIC and ort in VOCALIC):
                    raise Exception(f'Nonvocalic {ort}, {ph} in ' + id_string)
                if not (ph==ort or ph == VOWELS_TO_SEMI[ort]):
                    raise Exception(f'Mismatch {ort}, {ph} in ' + id_string)
                if ph in VOWELS:
                    vowel_count += 1
            if vowel_count != 1:
                raise Exception(f'Bad vowel count {vowel_count} in ' + id_string)

def _extend_clusters_stress() -> dict[str, str]:
    _clusters_sanity_check()
    stressed_clusters = {}
    for ortho, phon_list in VOWEL_CLUSTERS.items():
        ortho = list(ortho)
        phon_list = phon_list if isinstance(phon_list, list) else [phon_list]
        for phon in phon_list:
            phon = phon.split(' ')
            ortho_stress, phon_stress = [], []
            id_string = f'"{''.join(ortho)}", "{phon}"'
            for ort, ph in zip(ortho, phon):
                if ph in VOWELS: # this is the full vowel, it can be stressed
                    ortho_stress.append("'"+ort)
                    phon_stress.append("'"+ph)
                else:
                    ortho_stress.append(ort)
                    phon_stress.append(ph)
            stressed_clusters[''.join(ortho_stress)] = ' '.join(phon_stress)
    return stressed_clusters


VOWEL_CLUSTERS |= _extend_clusters_stress()

def preliminary_changes(s: str) -> str:
    return s.replace("â", "î")

def syllable_stress_form_to_phon(s : str) -> str:
    orig_s = s
    any_vowel = "[" + ''.join(VOWELS) + "'" + "]" # include stress mark
    s = preliminary_changes(s)
    syllables = s.split('.')
    new_syllables = []
    for syll_count, syllable in enumerate(syllables):
        # extract vowel clusters
        clusters = [cl for cl in re.split(f"({any_vowel}+)", syllable) if cl]
        vowel_cluster_count = 0
        new_clusters = []
        # new clusters will be space separated
        for cl_count, cl in enumerate(clusters):
            if not VOWELS.intersection(cl): # no vowels, it's consonants
                new_clusters.append(' '.join(cl))
            elif vowel_cluster_count > 0: # this is not the first vowel cluster
                # print(syll_count, len(syllables), cl_count, len(clusters), f'"{cl}"')
                if cl == "i" and syll_count == len(syllables)-1 and cl_count == len(clusters)-1:
                    # final short i
                    new_clusters.append(FINAL_I)
                else:
                    raise Exception(f'Extra vowel cluster {cl} in syll {syllable} in "{orig_s}"')
            else: # regular vowel cluster
                vowel_cluster_count += 1
                if cl not in VOWEL_CLUSTERS:
                    raise Exception(f"Unknown vowel cluster {cl} in {orig_s}")
                new_cluster = VOWEL_CLUSTERS[cl]
                if isinstance(new_cluster, list): # ambiguous
                    # no stress, this is 'i̯ u'
                    new_cluster = 'i̯ u' if cl == 'iu' else " | ".join(new_cluster)
                new_clusters.append(new_cluster) # is already space separated
        new_syllable = " ".join(new_clusters)
        new_syllables.append(new_syllable)
    return " . ".join(new_syllables)

def group_phon_sounds(s : str) -> str:
    # first group c(h?)[i̯e̯]
    groups = ['c e̯', 'g e̯', 'c h e̯', 'g h e̯', 'c i̯', 'g i̯', 'c h i̯', 'g h i̯']
    replacements = [g.replace(' ', '') for g in groups]
    for g,r in zip(groups, replacements):
        s = s.replace(g, r)
    # next group o̯a, e̯a
    # ea must be preceded by a space

    groups = ['o̯ a', "o̯ 'a", ' e̯ a', " e̯ 'a"]
    replacements = ['o̯a', "o̯'a", ' e̯a', " e̯'a"]
    for g,r in zip(groups, replacements):
        s = s.replace(g, r)

    return s

if __name__ == "__main__":
    import pandas as pd
    from edit_distance import get_edits

    df = pd.read_csv('./lang_data/change_records.doom.v2.work2.csv', sep='\t').fillna('')

    IS_OK = ['', 'OK', 'OK1', 'OK0']
    def get_phon_form(s) -> str:
        try:
            f = syllable_stress_form_to_phon(s)
            f = group_phon_sounds(f)
            return f
        except Exception as e:
            return ''

    df['input_phon'] = df.apply(lambda row: '' if row['error'] not in IS_OK
                                                else get_phon_form(row['input_syll']), axis=1)
    df['output_phon'] = df.apply(lambda row: '' if row['error'] not in IS_OK
                                                    else get_phon_form(row['output_syll']), axis=1)


    VOCALIC_ALL = VOCALIC | {FINAL_I, "'", '̯'}

    def cost_fn(i, o) -> int:
        if i == o:
            return 0
        if (i == '.' and o is not None) or (o == '.' and i is not None): # should be none!
            return 10
        if not isinstance(i, str) or not isinstance(o, str):
            return 1 # insertions/deletions
        is_vocalic = [set(s).issubset(VOCALIC_ALL) for s in (i, o)]
        if is_vocalic[0] == is_vocalic[1]:
            return 1
        return 10

    def count_final_blanks(chg_seq: list[tuple[Any, Any]]) -> int:
        i = 0
        while(chg_seq[len(chg_seq)-1-i][0] is None):
            i +=1
        return -i

    def get_change_seq(phon_in: str, phon_out: str) -> str:
        if not phon_out or not phon_in:
            return ''
        phon_in, phon_out = [[ph for ph in phon.split(' ')] for phon in (phon_in, phon_out)]
        edits = get_edits(phon_in, phon_out, cost_fn=cost_fn)
        if len(edits)>1:
            edits.sort(key=count_final_blanks)
        edit = edits[0]
        return ' '.join([f"{chg[0] if chg[0] else '0'}/{chg[1] if chg[1] else '0'}" for chg in edit])

    df['phon_changes'] = df.apply(lambda row: get_change_seq(row['input_phon'], row['output_phon']), axis=1)

        

        

