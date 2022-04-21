import global_vars


class Cycle:

    def __init__(self, cycle_id):

        self.cycle_id = cycle_id
        # self.discharge_dQdV = global_vars.batteries_data['cycles']['discharge_dQdV']
        self.t = global_vars.batteries_data['cycles'][self.cycle_id]['t']
        self.Qc = global_vars.batteries_data['cycles'][self.cycle_id]['Qc']
        self.I = global_vars.batteries_data['cycles'][self.cycle_id]['I']
        self.V = global_vars.batteries_data['cycles'][self.cycle_id]['V']
        self.T = global_vars.batteries_data['cycles'][self.cycle_id]['T']
        self.Qd = global_vars.batteries_data['cycles'][self.cycle_id]['Qd']
        self.Qdlin = global_vars.batteries_data['cycles'][self.cycle_id]['Qdlin']
        self.Tdlin = global_vars.batteries_data['cycles'][self.cycle_id]['Tdlin']

    def print(self):
        print("current cycle:")
        print(self.cycle_id)
        print("t:")
        print(self.t)
        print("Qc:")
        print(self.Qc)
        print("I:")
        print(self.I)
        print("V:")
        print(self.V)
        print("T:")
        print(self.T)
        print("Qd:")
        print(self.Qd)
        print("Qdlin:")
        print(self.Qdlin)
        print("Tdlin:")
        print(self.Tdlin)


