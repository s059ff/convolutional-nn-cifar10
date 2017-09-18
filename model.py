import chainer
import chainer.functions as F
import chainer.links as L

class CnvolutionalNN(chainer.Chain):

    def __init__(self):
        super(CnvolutionalNN, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(0.1)
            self.c1 = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, initialW=w)
            self.bn1 = L.BatchNormalization(32)
            self.p1 = F.max_pooling_2d
            self.c2 = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1, initialW=w)
            self.bn2 = L.BatchNormalization(64)
            self.p2 = F.max_pooling_2d
            self.c3 = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, initialW=w)
            self.bn3 = L.BatchNormalization(128)
            self.p3 = F.max_pooling_2d
            self.fc = L.Linear(128 * 4 * 4, 10)

    def __call__(self, x):
        h = x
        h = self.p1(F.relu(self.bn1(self.c1(h))), ksize=2)
        h = self.p2(F.relu(self.bn2(self.c2(h))), ksize=2)
        h = self.p3(F.relu(self.bn3(self.c3(h))), ksize=2)
        h = F.softmax(self.fc(F.dropout(h)))
        return h
