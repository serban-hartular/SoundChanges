import pandas as pd
import dataclasses

from lexicon_utils import ORTHO_VOWELS

def is_vowel(ch : str) -> bool:
    return ch in ORTHO_VOWELS

def is_consonant(ch : str) -> bool:
    return ch not in ORTHO_VOWELS

CONSONANT_ENDING = '0'

def ending_matches(form: str, ending: str) -> bool:
    if ending == CONSONANT_ENDING and form and is_consonant(form[-1]):
        return True
    return form.endswith(ending)

@dataclasses.dataclass
class NounFlexionEntry:
    gen: str
    term_sg: str
    term_pl: str
    desin_sg: str
    desin_pl: str
    type: str
    examples: str

    def __post_init__(self):
        # sanity check
        for des, term in zip([self.desin_sg, self.desin_pl], [self.term_sg, self.term_pl]):
            if not term.endswith(des):
                raise Exception(f'termination "{term}" does not end with desinence "{des}"!')

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d : dict) -> 'NounFlexionEntry':
        return NounFlexionEntry(**d)

    def matches(self, form_sg: str, form_pl: str) -> bool:
        return all([ending_matches(form, term)
                    for form, term in zip([form_sg, form_pl], [self.term_sg, self.term_pl])])

    def get_radices(self, form_sg: str, form_pl: str) -> tuple[str, str]:
        return tuple([form[:-len(desin)] if len(desin)>0 else form
                    for form, desin in zip([form_sg, form_pl], [self.desin_sg, self.desin_pl])
        ])    

def NFE_from_df(df : pd.DataFrame) -> list[NounFlexionEntry]:
    return [NounFlexionEntry.from_dict(d) for d in df.to_dict(orient='records')]

def filter_nfes(form_sg: str, form_pl:str, nfe_list : list[NounFlexionEntry]) -> NounFlexionEntry|None:
    nfe_list = [nfe for nfe in nfe_list if nfe.matches(form_sg, form_pl)]
    if len(nfe_list) < 2:
        return nfe_list[0] if nfe_list else None
    # este o consoana in desinenta de plural (eg -uri, -ele, chiar si -ale, -îni)?
    desinences_with_consonants = [nfe for nfe in nfe_list
        if len(nfe.desin_pl)>1 and is_consonant(nfe.desin_pl[-2]) and ( # si nu se afla la finele
            len(form_sg)>1 and nfe.desin_pl[-2] not in form_sg[-2:]) # radicalului de singular
    ]
    if desinences_with_consonants:
        if len(desinences_with_consonants)==1:
            return desinences_with_consonants[0]
        nfe_list = desinences_with_consonants 
    # sortam dupa asemanarea de lungimi a radacinilor
    nfe_and_radices = [(nfe,)+nfe.get_radices(form_sg, form_pl) for nfe in nfe_list]
    root_len_diff_min = min([abs(len(t[1])-len(t[2])) for t in nfe_and_radices])
    nfe_and_radices = [t for t in nfe_and_radices if abs(len(t[1])-len(t[2])) == root_len_diff_min]
    nfe_and_radices.sort(key=lambda t: min([len(t[1]), len(t[2])]))
    return nfe_and_radices[0][0]
      

if __name__ == "__main__":
    import pickle
    import itertools

    df = pd.read_csv('./lang_data/subst_flex_clase.tsv', sep='\t')
    df = df.fillna('')
    nfe_list = NFE_from_df(df)

    # print('Loading')
    # df = pd.read_csv('./lexicon/nouns.v3.tsv', sep='\t')
    with open('./lexicon/plurals_indices.p', 'rb') as handle:
        plural_dict : dict[tuple, list] = pickle.load(handle)

    pairs = {}
    for (_, form_sg), plur_list in plural_dict.items():
        for _, form_pl in plur_list:
            pairs[(form_sg, form_pl)] = list()
            for nfe in nfe_list:
                if nfe.matches(form_sg, form_pl):
                    pairs[(form_sg, form_pl)].append(nfe)


    pairs2 = {k:v for k,v in pairs.items() if len(v)==2}
    for k,v in pairs2.items():
        print(k, filter_nfes(k[0], k[1], v).get_radices(*k))

