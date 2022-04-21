from Lib import argparse

import reader
from battery.Battery import Battery


global batteries_data


if __name__ == "__main__":

    batteries_data = reader.all_batteries()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--barcode', type=str, default=batteries_data.keys()[0],
        help="",
    )
    parser.add_argument('-c', '--cycle', type=int, help="", required=True)
    args = parser.parse_args()

    battery = Battery(args.barcode)
    battery.get_cycle()
    battery.reset()
    battery.next_cycle()
    battery.roll_back()

