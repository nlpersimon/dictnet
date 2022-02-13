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
def reverse_dictionary(query: str, response_class=UJSONResponse):
    try:
        message = []
        for senset, similarity in sensenet.reverse_dictionary(query):
            senset_json = rebuild_senset(senset).to_json()
            for sense in senset_json['senses']:
                sense['highlight'] = get_highlight_indices(
                    sense['en_def'], query)
            message.append(
                {'senset': senset_json, 'similarity': round_sig(similarity)})
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


@app.get('/api/compound')
def parse_compound(compound: str, response_class=UJSONResponse):
    response = {'query': '', 'pos': ''}
    if compound:
        pos_part = re.findall('pos:.*$', compound)
        if pos_part:
            _, pos = pos_part[0].split(':')
            response['pos'] = pos
            definition = compound.replace(pos_part[0], '').strip()
        else:
            definition = compound.strip()
        response['query'] = definition
    return response
