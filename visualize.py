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
    cols = 10
    rows = len(W) // cols + (1 if len(W) % cols != 0 else 0)
    plt.figure(num=None, figsize=(cols, rows), dpi=30, facecolor='w', edgecolor='k')
    for i in range(len(W)):
        plt.subplot(rows, cols, i + 1)
        plt.tick_params(labelleft='off', top='off', bottom='off')
        plt.tick_params(labelbottom='off', left='off', right='off')
        plt.imshow(W[i].transpose(1, 2, 0))
    plt.savefig(fname)
    plt.close()
    