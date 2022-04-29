from torch import optim
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from data.load_data import load_data
from mdn.models import MixtureDensityNetwork
from loguru import logger


def plot_data(x, y):
    plt.hist2d(x, y, bins=35)
    plt.xlim(-6, 0)
    plt.ylim(0, 3000)
    plt.axis('off')


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--n-iterations", type=int, default=50000)
    args = argparser.parse_args()

    x, y = load_data()

    model = MixtureDensityNetwork(1, 1, n_components=3)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    for i in range(args.n_iterations):
        optimizer.zero_grad()
        loss = model.loss(x, y).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

    print(y[0:10])
    print("---------------")
    samples = model.sample(x)
    print(samples[0:10])
    plt.figure(figsize=(8, 3))
    # plt.subplot(1, 2, 1)
    plt.hist2d(x[:, 0].numpy(), y[:, 0].numpy(), bins=35)
    plt.xlim(-6, 0)
    plt.ylim(0, 3000)
    plt.axis('off')
    plt.title("Observed data")
    # plt.subplot(1, 2, 2)


    plt.figure(figsize=(8, 3))
    plt.hist2d(x[:, 0].numpy(), samples[:, 0].numpy(), bins=35)
    plt.xlim(-6, 0)
    plt.ylim(0, 5000)
    plt.axis('off')
    plt.title("Sampled data")
    plt.show()

