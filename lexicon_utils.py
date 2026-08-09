import dataclasses
import re
import pandas as pd

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


def stress2syllables(e : FormEntry, stress_sym : str = STRESS_MARK, syll_sym : str = SYLLABLE_MARK) -> str:
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

def group_vowel_clusters(in_syllables : str,
                         stress_sym : str = STRESS_MARK, syll_sym : str = SYLLABLE_MARK)\
                                                                     -> list[str]:
    syll_list = in_syllables.split(syll_sym)
    output_list = []
    to_cluster = ORTHO_VOWELS | {stress_sym}
    for syllable in syll_list:
        cluster_list = []
        for c in syllable:
            if c in to_cluster and cluster_list and cluster_list[-1][-1] in to_cluster:
                cluster_list[-1] += c
            else:
                cluster_list.append(c)
        output_list.extend(cluster_list)
    return output_list

def stress_mark_cleanup(in_clusters : list[str],
                        stress_sym : str = STRESS_MARK):
    """If there is no stress sybmol, add it to the first cluster.
    Move stress symbol to beginning of vowel cluster if needed.
    Operates in-place."""
    for i, cluster in enumerate(in_clusters):
        if stress_sym in cluster:
            in_clusters[i] = stress_sym + cluster.replace(stress_sym, '')
            return
    # no stress symbol found, add one
    for i, cluster in enumerate(in_clusters):
        if ORTHO_VOWELS.intersection(cluster): # this cluster has vowels
            in_clusters[i] = stress_sym + cluster
            return
    raise Exception('No vowel cluster in ' + str(in_clusters))


def get_entry_by_index(df : pd.DataFrame, index : int, INDEX_COL : str = 'INDEX') -> FormEntry:
    df = df[df[INDEX_COL]==index]
    if len(df) != 1:
        raise Exception(f'Search for index {index} yielded {len(df)} rows!')
    return FormEntry.from_dict(df.to_dict(orient='records')[0])

def index_to_letterclusters(df : pd.DataFrame, index : int) -> list[str]:
    e = get_entry_by_index(df, index)
    sylls_w_stress = stress2syllables(e)
    cluster_list = group_vowel_clusters(sylls_w_stress)
    stress_mark_cleanup(cluster_list)
    return cluster_list

def generate_root(cluster_list : list[str]) -> tuple[list[str], str]:
    """Remove final vowel. Not in place"""
    cluster_list = [s for s in cluster_list]
    desinence = ''
    if cluster_list and ORTHO_VOWELS.intersection(cluster_list[-1]): # last syllable has vowels
        last = cluster_list[-1][:-1] # remove last letter
        desinence = cluster_list[-1][-1]
        if not last: # nothing left
            cluster_list.pop()
        elif last == STRESS_MARK: # if this was solitary stressed vowel, don't remove it
            desinence = ''
        else: # replace final cluster with the one that has no vowel 
            cluster_list[-1] = last
    return cluster_list, desinence


if __name__ == "__main__":

    df = pd.read_csv('./lexicon/nouns.tsv', sep='\t')
    #df['include'] = df.apply(lambda row: filter_fn(row['xpos']), axis=1)
    #df = df[df['include']==1]
    


