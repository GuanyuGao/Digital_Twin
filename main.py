from Lib import argparse
import global_vars
from battery import Battery


if __name__ == "__main__":

    global_vars.init()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--barcode', type=str, default=global_vars.batteries_data.keys()[0],
        help="",
    )
    parser.add_argument('-c', '--cycle', type=int, help="", required=True)
    args = parser.parse_args()

    battery = Battery(args.barcode)
    battery.get_cycle()
    battery.reset()
    battery.next_cycle()
    battery.roll_back()

