from battery.Cycle import Cycle
from main import batteries_data


class Battery:

    def __init__(self, barcode):
        self.barcode = barcode
        self.cur_cycle = 0
        self.cycles = [Cycle(barcode, key) for key in batteries_data[barcode]['cycles'].keys()]

    def get_cycle(self, cycle_id=None):

        if cycle_id is None:
            return self.cycles[self.cur_cycle]
        else:
            return self.cycles[cycle_id]

    def next_cycle(self):
        self.cur_cycle += 1
        return self.cycles[self.cur_cycle]

    def reset(self):
        self.cur_cycle = 0

    def roll_back(self, cycle_id):
        self.cur_cycle = cycle_id

