import reader

global batteries_data


def init():
    batteries_data = reader.all_batteries()
