import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
import datetime
import numpy as np
import os
import pickle
import shutil
import tarfile
import urllib.request

from model import CnvolutionalNN
from visualize import visualize, visualize_kernel

# Define constants
N = 100     # Minibatch size
SNAPSHOT_INTERVAL = 10

def main():

    # (Make directories)
    os.mkdir('dataset/') if not os.path.isdir('dataset') else None
    os.mkdir('train/') if not os.path.isdir('train') else None

    # (Download dataset)
    if not os.path.exists('dataset/cifar-10-train-x.npy'):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/cifar-10-python.tar.gz', 'wb') as stream:
            stream.write(response.read())
        with tarfile.open('dataset/cifar-10-python.tar.gz', 'r') as stream:
            stream.extractall('dataset/')
        train_x = []
        train_y = []
        for path in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
            path = 'dataset/cifar-10-batches-py/' + path
            with open(path, 'rb') as stream:
                pair = pickle.load(stream, encoding='bytes')
                images = pair[b'data']
                labels = pair[b'labels']
                train_x.append(images)
                train_y.append(labels)
        train_x = np.array(train_x, dtype='f').reshape((-1, 3, 32, 32)) / 255.
        train_y = np.array(train_y, dtype='i')
        np.save('dataset/cifar-10-train-x', train_x)
        np.save('dataset/cifar-10-train-y', train_y)

        test_x = []
        test_y = []
        for path in ['test_batch']:
            path = 'dataset/cifar-10-batches-py/' + 'test_batch'
            with open(path, 'rb') as stream:
                pair = pickle.load(stream, encoding='bytes')
                images = pair[b'data']
                labels = pair[b'labels']
                test_x.append(images)
                test_y.append(labels)
        test_x = np.array(test_x, dtype='f').reshape((-1, 3, 32, 32)) / 255.
        test_y = np.array(test_y, dtype='i')
        np.save('dataset/cifar-10-test-x', test_x)
        np.save('dataset/cifar-10-test-y', test_y)
    # os.remove('dataset/cifar-10-python.tar.gz') if os.path.exists('dataset/cifar-10-python.tar.gz') else None
    # shutil.rmtree('dataset/cifar-10-batches-py', ignore_errors=True)

    # Create samples.
    train_x = np.load('dataset/cifar-10-train-x.npy').reshape((-1, 3, 32, 32))    # cifar-10 or cifar-100
    train_y = np.load('dataset/cifar-10-train-y.npy').reshape((-1))               # cifar-10 or cifar-100
    test_x = np.load('dataset/cifar-10-test-x.npy').reshape((-1, 3, 32, 32))      # cifar-10 or cifar-100
    test_y = np.load('dataset/cifar-10-test-y.npy').reshape((-1))                 # cifar-10 or cifar-100

    # Create the model
    nn = CnvolutionalNN()

    # (Use GPU)
    chainer.cuda.get_device(0).use()
    nn.to_gpu()

    # Setup optimizers
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(nn)

    # (Change directory)
    os.chdir('train/')
    time = datetime.datetime.today().strftime("%Y-%m-%d %H.%M.%S")
    os.mkdir(time)
    os.chdir(time)

    # Training
    for epoch in range(200):

        # (Validate generated images)
        if (epoch % SNAPSHOT_INTERVAL == 0):
            os.mkdir('%d' % epoch)
            os.chdir('%d' % epoch)
            visualize_kernel(chainer.cuda.to_cpu(nn.c0.W.data), 'kernel.png')
            os.chdir('..')

        # (Random shuffle samples)
        perm = np.random.permutation(len(train_x))

        total_loss_train = 0.0
        total_loss_test = 0.0

        for n in range(0, len(perm), N):
            x = chainer.cuda.to_gpu(train_x[perm[n:n + N]])
            t = chainer.cuda.to_gpu(train_y[perm[n:n + N]])
            y = nn(x)
            loss = F.softmax_cross_entropy(y, t)
            nn.cleargrads()
            loss.backward()
            optimizer.update()
            total_loss_train += loss.data

        x = chainer.cuda.to_gpu(test_x)
        t = chainer.cuda.to_gpu(test_y)
        y = nn(x)
        loss = F.softmax_cross_entropy(y, t)
        total_loss_test += loss.data

        # (View loss)
        total_loss_train /= len(perm) / N
        print(epoch, total_loss_train, total_loss_test)


if __name__ == '__main__':
    main()
