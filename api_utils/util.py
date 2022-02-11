from math import floor, log10
import os
import jsonlines
from view_objects.senset_view import SensetView


def load_additional_information():
    work_dir, _ = os.path.split(__file__)
    paraent_dir, _ = os.path.split(work_dir)
    additional_info_path = os.path.join(
        paraent_dir, 'data', 'v0.2.0', 'wn_bi-camb', 'additional_info.jsonl')
    with jsonlines.open(additional_info_path) as f:
        id_to_additional_info = {line['sense_id']: line for line in f}
    return id_to_additional_info


id_to_additional_info = load_additional_information()


def rebuild_senset(senset):
    camb_sense = find_camb_sense(senset.senses)
    level = guideword = ch_def = ''
    if camb_sense:
        additional_info = id_to_additional_info[camb_sense.sense_id]
        level = additional_info['level']
        guideword = additional_info['guideword']
        ch_def = additional_info['ch_definition']
    senset_view = SensetView(
        senset, level, guideword, ch_def)
    return senset_view


def find_camb_sense(senses):
    camb_sense = None
    for sense in senses:
        if sense.source == 'cambridge':
            camb_sense = sense
            break
    return camb_sense


def round_sig(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)
