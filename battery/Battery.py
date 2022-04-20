class Battery:

    def __init__(self, barcode):
        self.barcode = barcode
        self.cur_cycle = 0

    def get_cycle(self, cycle_id):
        pass

    def next_cycle(self, ):
        pass

    def reset(self):
        self.cur_cycle = 0

    def roll_back(self, cycle_id):
        self.cur_cycle = cycle_id

