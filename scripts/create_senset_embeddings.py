import argparse
import tqdm
import jsonlines
import numpy as np
import gensim
from gensim.models import KeyedVectors
from src.sensenet.schema.senset_file import SensetFileReader


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--senset_file', type=str, dest='senset_file',
                        help='path to the senset file')
    parser.add_argument('-e', '--sense_embeds', type=str, dest='sense_embeds',
                        help='path to the sense embeddings file')
    parser.add_argument('-o', '--output', type=str, dest='output',
                        help='path to the output senset embeddings file')
    return parser


def main():
    parser = prepare_parser()
    args = parser.parse_args()

    with jsonlines.open(args.senset_file) as f:
        sensets = list(SensetFileReader(f).read())

    sense_embeds = KeyedVectors.load_word2vec_format(args.sense_embeds,
                                                     binary=False)

    with gensim.utils.open(args.output, 'wb') as f:
        f.write(gensim.utils.to_utf8(f'{len(sensets)} {300}\n'))

        for senset in tqdm.tqdm(sensets):
            mean_embeds = np.mean(
                [sense_embeds[sense.sense_id]
                 for sense in senset.senses],
                axis=0)
            embeds_str = ' '.join([str(num) for num in mean_embeds])
            f.write(gensim.utils.to_utf8(f'{senset.senset_id} {embeds_str}\n'))


if __name__ == '__main__':
    main()
