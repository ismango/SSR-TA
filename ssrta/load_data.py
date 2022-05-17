from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import re
from string import digits
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wordnet
from tqdm import tqdm
import numpy as np


class TicketDataset(Dataset):
    def __init__(self, data, dtext, rtext, expert):
        self.data = data
        self.dtext = dtext
        self.rtext = rtext
        self.expert = expert

    def __getitem__(self, index):
        self.description = torch.LongTensor(self.dtext.transform(self.data['description'][index]))
        self.resolution = torch.LongTensor(self.rtext.transform(self.data['resolution'][index]))
        self.expert = torch.FloatTensor([self.expert.vocab.get(self.data['expert'][index])])[0]
        return self.description, self.resolution, self.expert

    def __len__(self):
        return len(self.data)


def load_data(train_data_path, test_data_path):
    train_data = data_clean(pd.read_csv(train_data_path))
    test_data = data_clean(pd.read_csv(test_data_path))

    dtext = Text()
    rtext = Text()
    expert = Expert()

    dtext.fit(train_data['description'])
    rtext.fit(train_data['resolution'])
    expert.fit(train_data['expert'])

    train_dataset = TicketDataset(train_data, dtext, rtext, expert)
    test_dataset = TicketDataset(test_data, dtext, rtext, expert)

    batch_size = 1024

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, test_loader, dtext, rtext, expert


class Text:
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'

    SOS = 0
    EOS = 1
    UNK = 2
    PAD = 3

    def __init__(self):
        self.vocab = {
            self.SOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS,
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }

        self.count = {}
        self.inverse_vocab = {}

    def fit(self, text, min=0, max=None, max_features=None):

        for sentence in text:
            for word in sentence.split():
                self.count[word] = self.count.get(word, 0) + 1
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.vocab[word] = len(self.vocab)

        self.inverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))

    def transform(self, sentence, max_len=20):

        sentence = sentence.split()
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.EOS_TAG] + [self.PAD_TAG] * (max_len - len(sentence))
            else:
                sentence = sentence[:max_len] + [self.EOS_TAG]
        sentence = [self.SOS_TAG] + sentence
        return [self.vocab.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        return [self.inverse_vocab.get(idx) for idx in indices]


class Expert:
    UNEXPERT_TAG = 'UNEXPERT'

    UNEXPERT = 0

    def __init__(self):
        self.vocab = {
            self.UNEXPERT_TAG: self.UNEXPERT
        }
        self.inverse_dict = {}

    def fit(self, experts):
        for expert in experts:
            if self.vocab.get(expert) is None:
                self.vocab[expert] = len(self.vocab)

        self.inverse_dict = dict(zip(self.vocab.values(), self.vocab.keys()))

    def transform(self, expert):
        expert_vec = [0] * len(self.vocab)
        expert_vec[self.vocab.get(expert)] = 1
        return expert_vec


f_stop = open('./data/stopwords_en.txt')
stop_list = [line.strip() for line in f_stop]
f_stop.close()

wnl = WordNetLemmatizer()


def data_clean(tickets):
    data = []
    for idx, ticket in tqdm(tickets.iterrows()):
        description = get_lemmatization(remove_stopwords(filter(ticket['description'])))
        resolution = get_lemmatization(remove_stopwords(filter(str(ticket['resolution']))))
        expert = ticket['expert'].lower()
        data.append([description, resolution, expert])
    data = pd.DataFrame(data, columns=['description', 'resolution', 'expert'])
    return data


def filter(text):
    cleanr = re.compile('<.*?>')
    r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    text = text.split('/')
    text = ' '.join(text)
    text = text.split('_')
    text = ' '.join(text)
    text = re.sub(cleanr, ' ', text)
    text = re.sub(r4, '', text)
    text = text.replace('-', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    return text


def remove_stopwords(text):
    res = []
    for word in text.strip().lower().split():
        if word not in stop_list:
            res.append(word)
    text = ' '.join(res)
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    return text


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag == 'CD':
        return 'CD'
    else:
        return None


def get_lemmatization(text):
    tokens = word_tokenize(text)
    tagged_sent = pos_tag(tokens)
    lemmas_sent = []
    for tag in tagged_sent:
        if tag[1] != 'CD':
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    return ' '.join(lemmas_sent)


def get_expert_dict(tickets):
    expert_dict_idx = {}
    expert_dict = {}
    for idx, ticket in tickets.iterrows():
        if ticket['expert'] not in expert_dict.values():
            expert_dict[len(expert_dict)] = ticket['expert']
        if expert_dict_idx.get(ticket['expert']) is None:
            expert_dict_idx[ticket['expert']] = len(expert_dict_idx)
    return expert_dict, expert_dict_idx


def get_expert_vec(expert, expert_dict_idx):
    expert_vec = list(np.zeros(len(expert_dict_idx)))
    expert_vec[expert_dict_idx.get(expert)] = 1
    return expert_vec
