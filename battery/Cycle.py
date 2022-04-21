from main import batteries_data


class Cycle:

    def __init__(self, barcode, key):

        self.cycle_id = int(key)
        self.discharge_dQdV = batteries_data[barcode]['cycles'][key]['discharge_dQdV']
        self.t = batteries_data[barcode]['cycles'][key]['t']
        self.Qc = batteries_data[barcode]['cycles'][key]['Qc']
        self.I = batteries_data[barcode]['cycles'][key]['I']
        self.V = batteries_data[barcode]['cycles'][key]['V']
        self.T = batteries_data[barcode]['cycles'][key]['T']
        self.Qd = batteries_data[barcode]['cycles'][key]['Qd']
        self.Qdlin = batteries_data[barcode]['cycles'][key]['Qdlin']
        self.Tdlin = batteries_data[barcode]['cycles'][key]['Tdlin']

    def get_cycle(self, cycle_id):

        return
