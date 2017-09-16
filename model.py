import chainer
import chainer.functions as F
import chainer.links as L

class CnvolutionalNN(chainer.Chain):

    def __init__(self):
        super(CnvolutionalNN, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, 30, ksize=3, stride=1, pad=1)
            self.p0 = F.max_pooling_2d
            self.c1 = L.Convolution2D(30, 60, ksize=3, stride=1, pad=1)
            self.p1 = F.max_pooling_2d
            self.c2 = L.Convolution2D(60, 120, ksize=3, stride=1, pad=1)
            self.p2 = F.max_pooling_2d
            self.l1 = L.Linear(120 * 5 * 5, 10)

    def __call__(self, x):
        h = x
        h = self.p0(F.relu(self.c0(h)), ksize=3, stride=2, pad=1)
        h = self.p1(F.relu(self.c1(h)), ksize=3, stride=2, pad=1)
        h = self.p2(F.relu(self.c2(h)), ksize=3, stride=2, pad=1)
        h = F.reshape(h, (-1, 120 * 5 * 5))
        h = F.softmax(self.l1(h))
        return h
