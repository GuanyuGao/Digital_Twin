from battery.Cycle import Cycle
import global_vars


class Battery:

    def __init__(self, barcode):
        self.barcode = barcode
        self.cur_cycle = 0
        self.cycles = [Cycle(key) for key in global_vars.batteries_data['cycles'].keys()]

    def get_cycle(self, cycle_id=None):

        if cycle_id is None:
            return self.cycles[self.cur_cycle].print()
        else:
            return self.cycles[cycle_id].print()

    def next_cycle(self):
        self.cur_cycle += 1
        return self.cycles[self.cur_cycle].print()

    def reset(self):
        self.cur_cycle = 0

    def roll_back(self, cycle_id):
        self.cur_cycle = cycle_id

