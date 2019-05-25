import os
import csv
import sys
import re
from cleaning import clean

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, word_list, target_list=None):
        self.guid = guid
        self.word_list = word_list
        self.target_list = target_list


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    # def get_labels(self):
    #     """Gets the list of labels for this data set."""
    #     raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                if len(line) == 2:
                    lines.append(line)
            return lines

class PersonalityProcessor(DataProcessor):
    def __init__(self, mode):
        self.mode = mode
        self.mode = self.mode.upper()

    def get_train_examples(self, data_dir, win_size):
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "train.csv")), "train", win_size)

    def get_dev_examples(self, data_dir, win_size):
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev", win_size)

    def get_vocab(self, data_dir, win_size):
        vocab_list = []
        train_examples = self.get_train_examples(data_dir, win_size)
        for i in train_examples:
            for word in i.word_list:
                if word not in vocab_list:
                    vocab_list.append(word)
        return vocab_list

    def create_examples(self, lines, set_type, win_size):
        examples = []
        word_list = []
        target_list = []

        for (i, line) in enumerate(lines):
            if (i == 0): continue
            id_num = "%s-%s" % (set_type, i)
            text = line[1]
            text = clean(text)
            text = text.split(" ")

            for i in range(int(len(text)/win_size)):
                word_list.append(text[i*win_size: (i+1)*win_size])
                target_list.append(text[i*win_size + 1: (i+1)*win_size + 1])

            examples.append(InputExample(guid=id_num, word_list=word_list, target_list=target_list))
        return examples
