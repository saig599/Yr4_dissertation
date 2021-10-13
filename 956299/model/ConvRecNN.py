import torch.nn as nn
import torch.nn.functional as f


class BiLSTM(nn.Module):
    # The Recurrent Neural Network
    def __init__(self, num_input, num_hidden, num_output):
        super(BiLSTM, self).__init__()

        self.rnnModel = nn.LSTM(num_input, num_hidden, bidirectional=True, num_layers=2)  # defining the BiLSTM module

        self.embedding = nn.Linear(num_hidden * 2, num_output)  # embedding for BiLSTM

    def forward(self, i):
        rnn, _ = self.rnnModel(i)
        T, b, h = rnn.size()
        t_rec = rnn.view(T * b, h)
        result = self.embedding(t_rec)
        result = result.view(T, b, -1)

        return result


class ConvRecNN(nn.Module):

    def __init__(self, img_height, num_channel, num_classes, lstm_hidden):
        super(ConvRecNN, self).__init__()

        conv_channels = [32, 32, 256, 256, 512, 720, 720, 720]  # Conv Layer input/output channel
        kernel_size = [3, 3, 3, 3, 3, 3, 3, 2]  # Kernel size
        stride_size = [1, 1, 1, 1, 1, 1, 1, 1]  # Stride size
        padding_size = [1, 1, 1, 1, 1, 1, 1, 0]  # Padding size

        cnnModel = nn.Sequential()

        def cnnNetwork(i, batchNorm=False):
            in_channel = num_channel if i == 0 else conv_channels[i - 1]  # compute input channel for each convolution
            out_channel = conv_channels[i]  # compute output channel for each convolution
            cnnModel.add_module('conv{0}'.format(i),
                                nn.Conv2d(in_channel, out_channel, kernel_size[i], stride_size[i], padding_size[i]))

            cnnModel.add_module('relu{0}'.format(i), nn.ReLU(True))  # Relu activation function

            if batchNorm:
                cnnModel.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(out_channel))  # batch normalization

        cnnNetwork(0)  # Relu for conv 1
        cnnModel.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # Maxpooling layer
        cnnNetwork(1)  # Relu for conv 2
        cnnModel.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # Maxpooling layer
        cnnNetwork(2, True)  # Relu for conv 3
        cnnNetwork(3)  # Relu for conv 4
        cnnModel.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # Maxpooling layer
        # cnn.add_module('dropout{0}'.format(1), nn.Dropout2d(0.2, False))  #drop out layer
        cnnNetwork(4, True)  # Relu for conv 5
        cnnNetwork(5)  # Relu for conv 6
        cnnNetwork(6)  # Relu for conv 7
        cnnModel.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # Maxpooling layer
        cnnNetwork(7, True)  # Relu for conv 8

        self.CNN = cnnModel  # call on the convolution neural network
        self.RNN = nn.Sequential(
            BiLSTM(720, lstm_hidden, lstm_hidden),  # BiLSTM layer
            BiLSTM(lstm_hidden, lstm_hidden, num_classes))  # BiLSTM layer

    def forward(self, i):
        # convolution features
        conv = self.CNN(i)
        batch, channel, height, width = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        # recurrent nn features
        result = self.RNN(conv)

        # log_softmax for converging output
        result = f.log_softmax(result, dim=2)

        return result
