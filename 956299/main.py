import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CTCLoss
import torch.nn as nn
import config
import os
import random
import engine
import dataset
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import model.ConvRecNN as model

# for storing accuracy and loss values
acc = []
trn_loss = []
val_loss = []

# custom weights initialization called on crnn
def weights_init(m):
    if type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# for make tests folder to store model checkpoints
if config.TEST is None:
    config.TEST = 'tests'
os.system('mkdir {0}'.format(config.TEST))

# for loading pre-trained model to continue training
if config.PRE_TRAINED != '':
    print('loading model from %s' % config.PRE_TRAINED)
    model.load_state_dict(torch.load(config.PRE_TRAINED), strict=False)

# To ensure random seed is the same everytime
random.seed(config.MANUAL_SEED)
np.random.seed(config.MANUAL_SEED)
torch.manual_seed(config.MANUAL_SEED)

# The train dataset and dataloader
train_dataset = dataset.CocotextDataset(root='./lmdb_train')

# the train data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE,
    num_workers=int(config.NUM_WORKERS),
    shuffle=True,
    collate_fn=dataset.cocoCollateFn(imgheight=config.IMG_HEIGHT,
                                     imgwidth=config.IMG_WIDTH,
                                     hold_ratio=config.KEEP_RATIO))

# the test dataset
val_dataset = dataset.CocotextDataset(
    root='./lmdb_val', transform=dataset.rescale((100, 32)))
# the test loader
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    shuffle=True,
    batch_size=config.BATCH_SIZE,
    num_workers=int(config.NUM_WORKERS))

# define number of class
num_class = len(config.ALPHABET) + 1

# convert between string and labels
convert = engine.converter(config.ALPHABET)

# define number of input channel
num_channel = 1

# Creating the model object and applying weights
model = model.ConvRecNN(config.IMG_HEIGHT, num_channel, num_class, config.H_LSTM)
model.apply(weights_init)
print(model)

# Calling CTC Loss
criterion = CTCLoss(blank=0, reduction='None')

length = torch.LongTensor(config.BATCH_SIZE)
text = torch.LongTensor(config.BATCH_SIZE * 5)
image = torch.FloatTensor(config.BATCH_SIZE, 3, config.IMG_HEIGHT, config.IMG_HEIGHT)

# moving model and data to gpu if cuda is available
if config.IS_CUDA:
    model.cuda()
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# defining model optimizer
optimizer = optim.Adam(model.parameters(), lr=config.L_RATE)

# computes average for torch.Tensor and torch.Variable
average_loss = engine.computeAverage()

# Dealing with nan for loss output when using CTC loss
if config.CTCNANLOSSERROR:
    if torch.__version__ >= '1.1.0':
        criterion = CTCLoss(zero_infinity=True)


# The model training function
def train(model, criterion, optimizer):
    model.train()

    data = train_iter.next()
    images, texts = data
    batch_size = images.size(0)

    engine.load(image, images)
    t, l = convert.encode(texts)
    engine.load(text, t)
    engine.load(length, l)
    optimizer.zero_grad()

    predictions = model(image)
    prediction_sizes = Variable(torch.LongTensor([predictions.size(0)] * batch_size))
    cost = criterion(predictions, text, prediction_sizes, length) / batch_size

    cost.backward()
    optimizer.step()

    return cost


# The model testing function
def val(model, criterion, iters=100):
    print('===============Starting OCR-Validation================')

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    average_loss = engine.computeAverage()

    iters = min(iters, len(val_loader))
    for i in range(iters):
        data = val_iter.next()
        i += 1
        images, texts = data
        batch_size = images.size(0)
        engine.load(image, images)
        t, l = convert.encode(texts)
        engine.load(text, t)
        engine.load(length, l)

        predictions = model(image)
        prediction_sizes = Variable(torch.LongTensor([predictions.size(0)] * batch_size))
        cost = criterion(predictions, text, prediction_sizes, length) / batch_size
        average_loss.add(cost)

        _, predictions = predictions.max(2)
        predictions = predictions.transpose(1, 0).contiguous().view(-1)
        final_predictions = convert.decode(predictions.data, prediction_sizes.data, raw=False)

        for prediction, target in zip(final_predictions, texts):
            if prediction == target.lower():
                n_correct += 1

    raw_predictions = convert.decode(predictions.data, prediction_sizes.data, raw=True)[:config.NUM_TEST_DISP]
    for raw_pred, prediction, ground_truth in zip(raw_predictions, final_predictions, texts):
        print('%-20s => %-20s, Ground Truth Text: %-20s' % (raw_pred, prediction, ground_truth))

    accuracy = (n_correct / float(iters * config.BATCH_SIZE)) * 100
    acc.append(accuracy)
    val_loss.append(average_loss.val())
    print('Test loss: %f, Model Accuracy: %f' % (average_loss.val(), accuracy))
    print('================OCR-Training================')


# The training and testing loop
for epoch in range(config.EPOCHS):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in model.parameters():
            p.requires_grad = True

        cost = train(model, criterion, optimizer)
        average_loss.add(cost)
        i += 1

        if i % config.DISP_INTERVAL == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, config.EPOCHS, i, len(train_loader), average_loss.val()))
            trn_loss.append(average_loss.val())
            average_loss.normalize()

        if i % config.VAL_INTERVAL == 0:
            val(model, criterion)

        # Saving checkpoint for each epoch of the model
        if i % config.SAVE_INTERVAL == 0:
            torch.save(
                model.state_dict(), '{0}/chkpt_{1}.pth'.format(config.TEST, epoch))

# generate some results
plt.plot(acc, label='Testing accuracy')
plt.style.use('ggplot')
plt.title('Model Accuracy')
plt.xlabel('Test Epochs')
plt.ylabel('Accuracy %')
plt.legend()
plt.show()

plt.plot(trn_loss, label='Training Loss')
plt.plot(val_loss, label='Testing Loss')
plt.style.use('ggplot')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
