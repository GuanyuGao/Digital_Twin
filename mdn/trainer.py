import torch
import json
from torch import optim
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from data.load_data import load_data, get_set
from mdn.models import MixtureDensityNetwork
from loguru import logger


def plot_data(x, y):
    plt.hist2d(x, y, bins=35)
    plt.xlim(-10, 0)
    plt.ylim(0, 500)
    plt.axis('off')


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--n-iterations", type=int, default=1000)
    args = argparser.parse_args()

    x, y = load_data()

    train_x, train_y = get_set(x, y, "train")
    test_x, test_y = get_set(x, y, "test")

    model = MixtureDensityNetwork(1, 1, n_components=2)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    for i in range(args.n_iterations):
        optimizer.zero_grad()
        loss = model.loss(train_x, train_y).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

    predictions = model.sample(test_x)
    json_type = {
        "test_x": test_x[:, 0].numpy().tolist(),
        "test_y": test_y[:, 0].numpy().tolist(),
        "predictions": predictions[:, 0].numpy().tolist()
    }
    with open('prediction.json', 'w') as f:
        json.dump(json_type, f)

    accuracy = torch.sum(abs(predictions[:, 0] - test_y[:, 0]) / test_y[:, 0]) / test_y.shape[0] * 100
    print(accuracy)
    # plt.figure(figsize=(8, 3))
    #
    # plt.subplot(1, 2, 1)
    # plot_data(x[:, 0].numpy(), y[:, 0].numpy())
    # plt.title("Observed data")
    #
    # plt.subplot(1, 2, 2)
    # plot_data(x[:, 0].numpy(), samples[:, 0].numpy())
    # plt.title("Sampled data")
    #
    # plt.show()



