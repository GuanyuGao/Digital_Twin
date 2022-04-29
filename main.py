import argparse
import global_vars
import reader
import json
from battery import Battery


batch_1 = './data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
batch_2 = './data/2017-06-30_batchdata_updated_struct_errorcorrect.mat'
batch_3 = './data/2018-04-12_batchdata_updated_struct_errorcorrect.mat'


if __name__ == "__main__":

    # global_vars.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--barcode', type=int, default=1, help="")
    parser.add_argument('-c', '--cycle', type=int, help="", required=True)
    args = parser.parse_args()
    global_vars.batteries_data = reader.load_battery(batch_1, args.barcode)
    # print(global_vars.batteries_data.keys())
    battery = Battery(args.barcode)

    json.loads()
    # battery.get_cycle()
    # battery.reset()
    # battery.next_cycle()
    # battery.roll_back(50)

