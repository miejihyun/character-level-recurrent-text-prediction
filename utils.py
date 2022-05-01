import torch
from torch.autograd import Variable

def batch_generator(x, batch_size):
    '''
    x: [num_words, in_channel, height, width]
    partitions x into batches
    '''
    num_step = x.size()[0] // batch_size
    for t in range(num_step):
        yield x[t*batch_size:(t+1)*batch_size]

def text2vec(words, char_dict, max_word_len):
    """ 
    Return list of list of int, in shape (len(words), max_word_len + 2)
    """
    word_vec = []
    for word in words:
        vec = [char_dict[ch] for ch in word] 
        if len(vec) < max_word_len:
            vec += [char_dict["PAD"] for _ in range(max_word_len - len(vec))]
        vec = [char_dict["BOW"]] + vec + [char_dict["EOW"]]
        word_vec.append(vec)
    return word_vec

def read_data(file_name):
    '''
    Return: list of strings (list of words)
    '''
    with open(file_name, 'r') as f:
        corpus = f.read().lower()
    import re
    corpus = re.sub(r"<unk>", "unk", corpus)
    return corpus.split()

def get_char_dict(vocabulary):
    '''
    vocabulary == dict of (word, int)
    Return: dict of (char, int), starting from 1
    '''
    char_dict = dict()
    count = 1
    for word in vocabulary:
        for ch in word:
            if ch not in char_dict:
                char_dict[ch] = count
                count += 1
    return char_dict

def create_word_char_dict(*file_name):
    '''
    Return a word dict and a char dict.
    '''
    text = []
    for file in file_name:
        text += read_data(file)
    word_dict = {word:ix for ix, word in enumerate(set(text))}
    char_dict = get_char_dict(word_dict)
    return word_dict, char_dict

def to_var(x):
    '''
    Return the tensor x with device cuda
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

