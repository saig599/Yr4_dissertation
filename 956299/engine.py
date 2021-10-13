import torch
import torch.nn as nn
import collections
from torch.autograd import Variable


# assure img ratio validity
def checkValidRatio(image):
    batch, channel, height, width = image.size()
    if height > width:
        upsampleImg = nn.UpsamplingBilinear2d(size=(height, height))
        image = upsampleImg(image)
    return image


class converter(object):
    def __init__(self, characters, non_cap=True):
        self.non_cap = non_cap
        if self.non_cap:
            characters = characters.lower()
        self.alphabet = characters + '-'
        self.dictionary = {}

        for i, char in enumerate(characters):
            self.dictionary[char] = i + 1

    def encode(self, text):
        # returns the encoded text and length of each text for CTC
        if type(text) == str:
            text = [self.dictionary[char.lower() if self.non_cap else char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = "".join(text)
            text, _ = self.encode(text)
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, text,length, raw=True):
        # returns decoded text or list of text for CTC conversion
        if length.numel() == 1:

            length = length[0]

            if not raw:
                return ''.join([self.alphabet[i - 1] for i in text])

            else:

                chars = []
                for i in range(length):
                    if text[i] != 0 and (not (i > 0 and text[i - 1] == text[i])):
                        chars.append(self.alphabet[text[i] - 1])
                return ''.join(chars)
        else:

            item = 0
            text_list = []

            for i in range(length.numel()):
                l = length[i]
                text_list.append(
                    self.decode(
                        text[item:item + l], torch.IntTensor([l]), raw=not raw))
                item += l
            return text_list


class computeAverage(object):
    # for computing average of torch.Tensor.

    def __init__(self):
        self.normalize()

    def add(self, v):
        if type(v) == Variable:

            count = v.data.numel()
            v = v.data.sum()

        elif type(v) == torch.Tensor:
            count = v.numel()
            v = v.sum()

        self.total += v

        self.num_count += count


    def normalize(self):
        self.total = 0
        self.num_count = 0


    def val(self):
        resource = 0
        if self.num_count != 0:
            resource = self.total / float(self.num_count)
        return resource


# load text and length tensors
def load(v, d):
    with torch.no_grad():
        v.resize_(d.size()).copy_(d)


