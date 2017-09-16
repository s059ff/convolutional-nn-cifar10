import matplotlib.pyplot as plt


def visualize(X, fname, shape):
    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    for i in range(0, 100):
        plt.subplot(10, 10, i + 1)
        plt.tick_params(labelleft='off', top='off', bottom='off')
        plt.tick_params(labelbottom='off', left='off', right='off')
        plt.imshow(X[i].reshape(shape).transpose(1, 2, 0))
    plt.savefig(fname)
    plt.close()

def visualize_kernel(W, fname):
    plt.figure(num=None, figsize=(10, 3), dpi=30, facecolor='w', edgecolor='k')
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.tick_params(labelleft='off', top='off', bottom='off')
        plt.tick_params(labelbottom='off', left='off', right='off')
        plt.imshow(W[i].reshape((3, 3, 3)).transpose(1, 2, 0))
    plt.savefig(fname)
    plt.close()
    