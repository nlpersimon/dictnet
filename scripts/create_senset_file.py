import argparse
import jsonlines
from gensim.models import KeyedVectors
from src.sensenet.schema.sense_file import SenseFileReader
from src.sensenet.schema.senset_file import SensetFileWriter
from src.sensenet.sas.sensets_extractor import SensetsExtractor


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sense_file', type=str, dest='sense_file',
                        help='path to the sense file')
    parser.add_argument('-e', '--sense_embeds', type=str, dest='sense_embeds',
                        help='path to the sense embeddings file')
    parser.add_argument('-o', '--output', type=str, dest='output',
                        help='path to the output senset file')
    return parser


def main():
    parser = prepare_parser()
    args = parser.parse_args()

    sense_embeds = KeyedVectors.load_word2vec_format(
        args.sense_embeds,
        binary=False
    )

    with jsonlines.open(args.sense_file) as f:
        senses = list(SenseFileReader(f).read())

    sensets_extractor = SensetsExtractor(senses, sense_embeds)
    all_sensets = sensets_extractor.all_sensets()

    with jsonlines.open(args.output, 'w') as f:
        writer = SensetFileWriter(f)
        for _, sensets in all_sensets.items():
            for senset in sensets:
                writer.write(senset)


if __name__ == '__main__':
    main()
