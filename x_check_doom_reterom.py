import pandas as pd
import pickle

def sing_stress_match(chg_rec : dict, doom_rec : dict) -> bool:
    if doom_rec['stress'] == '':
        return True
    return chg_rec['input_syll'].replace(".", "").find("'") == doom_rec['stress']

def plur_form_match(chg_rec : dict, doom_rec : dict) -> bool:
    return chg_rec['output_form'] == doom_rec['plural']

def plur_stress_match(chg_rec : dict, doom_rec : dict) -> bool:
    if doom_rec['plural_stress'] == "":
        return True
    return chg_rec['output_syll'].replace(".", "").find("'") == doom_rec['plural_stress']

def sing_syll_match(chg_rec : dict, doom_rec : dict) -> bool:
    if not doom_rec['syll_sg']:
        return True
    syll_sg = [s.strip() for s in doom_rec['syll_sg'].split('/')]
    return any([s.replace('-', '.') in chg_rec['input_syll'].replace("'", "") for s in syll_sg])

def plur_syll_match(chg_rec : dict, doom_rec : dict) -> bool:
    if not doom_rec['syll_pl']:
        return True
    syll_pl = [s.strip() for s in doom_rec['syll_pl'].split('/')]
    return any([s.replace('-', '.') in chg_rec['output_syll'].replace("'", "") for s in syll_pl])

test_fns = [
    ('sing_stress_match', sing_stress_match, 'stress'),
    ('plur_form_match', plur_form_match, 'plural'),
    ('plur_stress_match', plur_stress_match, 'plural_stress'),
    ('sing_syll_match', sing_syll_match, 'syll_sg'),
    ('plur_syll_match', plur_syll_match, 'syll_pl'),
]

def check_change_record(ch_rec: dict, doom_rec_list : list[dict]) -> tuple[bool, str, str]:
    for fn_name, test_fn, doom_attrib in test_fns:
        old_list = doom_rec_list
        doom_rec_list = [doom_rec for doom_rec in doom_rec_list if test_fn(ch_rec, doom_rec)]
        if not doom_rec_list:
            doom_params = ' | '.join([str(dr[doom_attrib]) for dr in old_list])
            return False, fn_name, doom_params
    return True, '', ''

if __name__ == "__main__":

    with open('./lang_data/doom3.nouns.both.dict.p', 'rb') as handle:
        doom_dict = pickle.load(handle)

    chg_data = pd.read_csv('./lang_data/change_records.doom.tsv', sep='\t').to_dict(orient='records')
    chg_data = [d for d in chg_data if d['in_DOOM3']]
    bad_data = []
    for chg_rec in chg_data:
        pass_flag, fn_name, param = check_change_record(chg_rec, doom_dict[chg_rec['input_form']])
        if not pass_flag:
            chg_rec.update({'error':fn_name, 'param':param})

    pd.DataFrame(chg_data).to_csv('./lang_data/change_records.doom.v2.tsv', sep='\t')



