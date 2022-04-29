import os
import json
import torch


def load_data():
    data_file = os.path.join(os.path.dirname(__file__), "./data1.json")
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


if __name__ == '__main__':
    load_data()
