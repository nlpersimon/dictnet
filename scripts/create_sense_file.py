import argparse
import jsonlines
from src.sensenet.prep.preprocessor import Preprocessor
from src.sensenet.prep.mappers.wordnet_mapper import WordnetMapper
from src.sensenet.prep.mappers.cambridge_mapper import CambridgeMapper
from src.sensenet.schema.sense_file import SenseFileWriter


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cambridge_file', type=str, dest='camb_file',
                        help='path to the cambridge dictionary file')
    parser.add_argument('-o', '--output', type=str, dest='output',
                        help='path to the output sense file')
    return parser


def main():
    parser = prepare_parser()
    args = parser.parse_args()

    wn_mapper = WordnetMapper()
    camb_mapper = CambridgeMapper()

    camb_file = jsonlines.open(args.camb_file)
    mappers = [wn_mapper.read(None), camb_mapper.read(camb_file)]

    preprocessor = Preprocessor()
    with jsonlines.open(args.output, 'w') as f:
        writer = SenseFileWriter(f)
        for mapper in mappers:
            for sense_file_line in preprocessor.preprocess(mapper):
                writer.write(sense_file_line)

    camb_file.close()


if __name__ == '__main__':
    main()
