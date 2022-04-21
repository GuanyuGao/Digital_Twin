import global_vars


class Cycle:

    def __init__(self, barcode, key):

        self.cycle_id = int(key)
        self.discharge_dQdV = global_vars.batteries_data[barcode]['cycles'][key]['discharge_dQdV']
        self.t = global_vars.batteries_data[barcode]['cycles'][key]['t']
        self.Qc = global_vars.batteries_data[barcode]['cycles'][key]['Qc']
        self.I = global_vars.batteries_data[barcode]['cycles'][key]['I']
        self.V = global_vars.batteries_data[barcode]['cycles'][key]['V']
        self.T = global_vars.batteries_data[barcode]['cycles'][key]['T']
        self.Qd = global_vars.batteries_data[barcode]['cycles'][key]['Qd']
        self.Qdlin = global_vars.batteries_data[barcode]['cycles'][key]['Qdlin']
        self.Tdlin = global_vars.batteries_data[barcode]['cycles'][key]['Tdlin']

    # def get_cycle(self, cycle_id):
    #
    #     return
