import jsonlines
from src.sensenet.prep.preprocessor import Preprocessor
from src.sensenet.prep.mappers.wordnet_mapper import WordnetMapper
from src.sensenet.prep.mappers.cambridge_mapper import CambridgeMapper
from src.sensenet.schema.sense_file import SenseFileWriter


def main():
    wn_mapper = WordnetMapper()
    camb_mapper = CambridgeMapper()

    camb_file = jsonlines.open('data/cambridge/cambridge.sense.000.jsonl')
    mappers = [wn_mapper.read(None), camb_mapper.read(camb_file)]

    preprocessor = Preprocessor()
    with jsonlines.open('data/v0.0.1/sense_file_bi-camb.jsonl', 'w') as f:
        writer = SenseFileWriter(f)
        for mapper in mappers:
            for sense_file_line in preprocessor.preprocess(mapper):
                writer.write(sense_file_line)

    camb_file.close()


if __name__ == '__main__':
    main()
