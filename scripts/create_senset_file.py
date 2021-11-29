import jsonlines
from gensim.models import KeyedVectors
from src.sensenet.schema.sense_file import SenseFileReader
from src.sensenet.schema.senset_file import SensetFileWriter
from src.sensenet.sas.sensets_extractor import SensetsExtractor


def main():
    sense_embeds = KeyedVectors.load_word2vec_format(
        'src/sensenet/ses/embeddings/wn_bi-camb_cpae.txt',
        binary=False
    )

    with jsonlines.open('data/v0.0.1/sense_file_bi-camb.jsonl') as f:
        senses = list(SenseFileReader(f).read())

    sensets_extractor = SensetsExtractor(senses, sense_embeds)
    all_sensets = sensets_extractor.all_sensets()
    # all_sensets = sensets_extractor.sensets('apple', 'NOUN')

    with jsonlines.open('data/v0.0.1/senset_file_bi-camb.jsonl', 'w') as f:
        writer = SensetFileWriter(f)
        for _, sensets in all_sensets.items():
            # for senset in all_sensets:
            for senset in sensets:
                writer.write(senset)


if __name__ == '__main__':
    main()
