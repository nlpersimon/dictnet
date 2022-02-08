from itertools import product

def disambiguate(sensenet, word1, word2):
    sensets1 = sensenet.sensets(word1)
    sensets2 = sensenet.sensets(word2)
    senset1, senset2 = max(
        product(sensets1, sensets2),
        key=lambda x: sensenet.senset_similarity(x[0].senset_id, x[1].senset_id))
    similarity = sensenet.senset_similarity(
        senset1.senset_id, senset2.senset_id)
    return ((senset1, senset2), similarity)
