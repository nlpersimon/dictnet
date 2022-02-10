from fastapi import FastAPI
from fastapi.responses import UJSONResponse
from starlette.middleware.cors import CORSMiddleware

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
            message.append({'senset': rebuild_senset(
                senset).to_json(), 'similarity': round_sig(similarity)})
    except Exception as e:
        message = str(e)
    return {'message': message}
