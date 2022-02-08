from math import floor, log10
from src.sensenet.schema.senset_file import Senset


def rebuild_senset(senset):
    return Senset(senset_id=senset.senset_id, word=senset.word, pos_norm=senset.pos_norm, senses=senset.senses)


def round_sig(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)
