import os
import json
import random
import torch


def load_data():
    data_file = os.path.join(os.path.dirname(__file__), "data.json")
    file = open(data_file, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)

    x = torch.tensor(papers[0]).t()
    x = torch.unsqueeze(x, dim=1)

    y = torch.tensor(papers[1]).t()
    y = torch.unsqueeze(y, dim=1)

    return x, y


def get_set(x, y, mode):

    length = x.shape[0]

    if mode == "train":
        set_len = length * 0.7
    elif mode == "test":
        set_len = length * 0.3

    x_set = torch.empty(0, 1)
    y_set = torch.empty(0, 1)

    for i in range(int(set_len)):
        rdt = random.randint(0, int(set_len) - 1)
        x_set = torch.cat([x_set, torch.unsqueeze(x[rdt], dim=1)], dim=0)
        y_set = torch.cat([y_set, torch.unsqueeze(y[rdt], dim=1)], dim=0)

    return x_set, y_set


if __name__ == '__main__':
    load_data()
