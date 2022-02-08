import argparse
import jsonlines
from _jsonnet import evaluate_file
import json
from src.sensenet.prep.preprocessor import Preprocessor
from src.sensenet.prep.mappers.wordnet_mapper import WordnetMapper
from src.sensenet.prep.mappers.cambridge_mapper import CambridgeMapper
from src.sensenet.prep.mappers.oxford_mapper import OxfordMapper
from src.sensenet.schema.sense_file import SenseFileWriter


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_path', metavar='settings_path', type=str)
    return parser


def set_mappers(dictionary_settings):
    indent_space = ' ' * 4
    mappers = []
    print('dictionaries:')
    for dictionary, data_path in dictionary_settings.items():
        print(f'{indent_space} {dictionary}: {data_path}')
        mapper_cls = dictionary_to_mapper[dictionary]
        if dictionary == 'wordnet':
            mappers.append(mapper_cls().read(None))
        elif dictionary == 'cambridge':
            camb_file = jsonlines.open(data_path)
            mappers.append(mapper_cls().read(camb_file))
        elif dictionary == 'oxford':
            oxford_file = open(data_path)
            mappers.append(mapper_cls().read(oxford_file))
        else:
            raise ValueError(
                f'{dictionary} is not supported. The system currently supports {list(dictionary_to_mapper.keys())}')
    return mappers


def set_preprocessor(preprocessor_settings):
    tokenize = preprocessor_settings.get('tokenize', True)
    remove_mwe = preprocessor_settings.get('remove_mwe', True)
    remove_stop_words = preprocessor_settings.get('remove_stop_words', True)
    indent_space = ' ' * 4
    print('Preprocessor with:')
    print(f'{indent_space} tokenize: {tokenize}')
    print(f'{indent_space} remove_mwe: {remove_mwe}')
    print(f'{indent_space} remove_stop_words: {remove_stop_words}')
    preprocessor = Preprocessor(
        tokenize=tokenize,
        remove_mwe=remove_mwe,
        remove_stop_words=remove_stop_words
    )
    return preprocessor


dictionary_to_mapper = {
    'wordnet': WordnetMapper,
    'cambridge': CambridgeMapper,
    'oxford': OxfordMapper
}


def main():
    parser = prepare_parser()
    args = parser.parse_args()

    settings = json.loads(evaluate_file(args.settings_path))

    mappers = set_mappers(settings['dictionary'])
    preprocessor = set_preprocessor(settings.get('preprocessor', {}))
    with jsonlines.open(settings['output_path'], 'w') as f:
        writer = SenseFileWriter(f)
        for mapper in mappers:
            for sense_file_line in preprocessor.preprocess(mapper):
                writer.write(sense_file_line)

    return


if __name__ == '__main__':
    main()
