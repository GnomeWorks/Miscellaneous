import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist, cifar10
from keras.optimizers import Adam
from keras import backend as K

from keras import initializers

IMAGE_SIZE = 28
#IMAGE_SIZE = 32

#K.set_image_dim_ordering('th')

np.random.seed(0xdedede)

randomDim = 100

net_counter = 0

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(60000, 784)

#(X_train, y_train), (X_test, y_test) = cifar10.load_data()

adam = Adam(lr=0.00015, beta_1=0.85)

generator = Sequential()
generator.add(Dense(256, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=.02)))
generator.add(LeakyReLU(.2))
generator.add(Dense(512))
generator.add(LeakyReLU(.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(.2))
generator.add(Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)


def buildGenerator(layerSize=256, leak=.2):
    global generator
    generator = Sequential()
    generator.add(Dense(layerSize, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=.02)))
    generator.add(LeakyReLU(leak))
    generator.add(Dense(layerSize * 2))
    generator.add(LeakyReLU(leak))
    generator.add(Dense(layerSize * 4))
    generator.add(LeakyReLU(leak))
    generator.add(Dense(IMAGE_SIZE*IMAGE_SIZE, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam)


discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=.02)))
discriminator.add(LeakyReLU(.2))
discriminator.add(Dropout(.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(.2))
discriminator.add(Dropout(.25))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(.2))
discriminator.add(Dropout(.2))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)


def buildDiscriminator(layerSize=1024, leak=.2, drop=.3, dropReduce=.05):
    global discriminator
    discriminator = Sequential()
    discriminator.add(Dense(layerSize, input_dim=IMAGE_SIZE*IMAGE_SIZE, kernel_initializer=initializers.RandomNormal(stddev=.02)))
    discriminator.add(LeakyReLU(.2))
    discriminator.add(Dropout(drop))
    discriminator.add(Dense(int(layerSize / 2)))
    discriminator.add(LeakyReLU(leak))
    discriminator.add(Dropout(drop - dropReduce))
    discriminator.add(Dense(int(layerSize / 4)))
    discriminator.add(LeakyReLU(leak))
    discriminator.add(Dropout(drop - (dropReduce * 2)))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)


# combine the two
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)


def combineNetworks():
    discriminator.trainable = False
    global ganInput
    ganInput = Input(shape=(randomDim,))
    x = generator(ganInput)
    global ganOutput
    ganOutput = discriminator(x)
    global gan
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adam)


dLosses = []
gLosses = []


def plotLoss(epoch):
    global dLosses
    global gLosses
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminator Loss')
    plt.plot(gLosses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./newimg/gan%d_loss_epoch_%d.png' % (net_counter, epoch))
    dLosses = []
    gLosses = []


def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, IMAGE_SIZE, IMAGE_SIZE)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('./newimg/gan_gen%d_img_epoch_%d.png' % (net_counter, epoch))


def saveModels(epoch):
    generator.save('./newmod/gan_gen%d_epoch_%d.h5' % (net_counter, epoch))
    discriminator.save('./newmod/gan_dis%d_epoch_%d.h5' % (net_counter, epoch))


def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] / batchSize
    print('Epochs: ', epochs)
    print('Batch Size: ', batchSize)
    print('Batches/Epoch: ', batchCount)

    for e in range(1, epochs + 1):
        print('\n', '-' * 15, 'Epoch %d' % e, '-' * 15)

        for _ in tqdm(range(0, int(batchCount))):
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            generatedImages = generator.predict(noise)

            X = np.concatenate([imageBatch, generatedImages])

            yDis = np.zeros(2 * batchSize)
            yDis[:batchSize] = 0.9

            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False

            gloss = gan.train_on_batch(noise, yGen)

        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 20 == 0:
            plotGeneratedImages(e)
            saveModels(e)

    plotLoss(e)


if __name__ == '__main__':
    # buildGenerator(256, .2)
    # buildDiscriminator(1024, .2, .3, .5)
    # combineNetworks()
    # train(200, 128)

    net_counter = 0

    buildGenerator(256, .2)
    buildDiscriminator(1024, .2, .3, .05)
    combineNetworks()
    train(200, 128)

    '''
    for i in range(1, 4):
        for j in np.arange(0.1, 1.0, .15):
            for k in np.arange(.05, .4, .05):
                for m in np.arange(k / 2, k, k / 4):
                    buildGenerator(256 * i, j)
                    buildDiscriminator(1024 * i, j, k, m)
                    combineNetworks()
                    train(200, 128)

                    net_counter += 1
    '''
