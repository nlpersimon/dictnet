from fastapi import FastAPI
from fastapi.responses import UJSONResponse
from starlette.middleware.cors import CORSMiddleware
import re

from src.sensenet.sensenet import load_sensenet
from api_utils.util import rebuild_senset, round_sig
from api_utils.gsd import disambiguate


sensenet = load_sensenet('data/v0.2.0/wn_bi-camb',
                         sensenet_type='mean')


app = FastAPI()

origins = ["*"]

# 設置跨域傳參
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/api/gsd')
def disambiguate_group_sense(w1: str, w2: str, response_class=UJSONResponse):
    try:
        (senset1, senset2), similarity = disambiguate(sensenet, w1, w2)
        senset1 = rebuild_senset(senset1)
        senset2 = rebuild_senset(senset2)
        message = {'senset1': senset1.to_json(), 'senset2': senset2.to_json(),
                   'similarity': round_sig(float(similarity), 2)}
    except Exception as e:
        message = str(e)
    return {'message': message}


@app.get('/api/rd')
def reverse_dictionary(query: str, pos: str = '', response_class=UJSONResponse):
    try:
        pos = pos.upper() or None
        message = []
        for senset, similarity in sensenet.reverse_dictionary(query, pos):
            senset_json = rebuild_senset(senset).to_json()
            for sense in senset_json['senses']:
                sense['highlight'] = get_highlight_indices(
                    sense['en_def'], query)
            message.append(
                {'senset': senset_json, 'similarity': round_sig(similarity)})
        message.sort(
            key=lambda msg: sum(len(sense['highlight']) for sense in msg['senset']['senses']), reverse=True)
    except Exception as e:
        message = str(e)
    return {'message': message}


def get_highlight_indices(definition, query):
    definition_tokens = definition.split()
    indices = []
    for query_token in query.split():
        for i, def_token in enumerate(definition_tokens):
            if def_token.startswith(query_token):
                indices.append(i)
                break
    return indices


@app.get('/api/senset')
def get_sensets(headword: str, pos: str = '', response_class=UJSONResponse):
    try:
        pos = pos.upper() or None
        message = []
        for senset in sensenet.sensets(headword, pos):
            senset_json = rebuild_senset(senset).to_json()
            message.append(
                {'senset': senset_json})
    except Exception as e:
        message = str(e)
    return {'message': message}


@app.get('/api/compound')
def parse_compound(compound: str, response_class=UJSONResponse):
    response = {'query': '', 'pos': '', 'senset': ''}
    if compound:
        pos_part = re.findall('pos:[a-zA-Z]+', compound)
        senset_part = re.findall('senset:[a-zA-Z]+', compound)
        compound, pos = split_compound(compound, pos_part)
        compound, senset = split_compound(compound, senset_part)
        response['pos'] = pos
        response['senset'] = senset
        response['query'] = compound
    return response


def split_compound(compound, part):
    param = ''
    if part:
        _, param = part[0].split(':')
        compound = compound.replace(part[0], '').strip()
    else:
        compound = compound.strip()
    return (compound, param)
