import h5py
import numpy as np


def load_battery(matFilename, barcode):

    f = h5py.File(matFilename)
    batch = f['batch']
    num_cells = batch['summary'].shape[0]
    battery = {}
    for i in range(num_cells):

        # barcode_ = f[batch['barcode'][i, 0]].value.tobytes()[::2].decode()
        # policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
        # channel_id = f[batch['channel_id'][i, 0]].value.tobytes()[::2].decode()
        # cl = f[batch['cycle_life'][i, 0]].value

        if barcode == i:
            cl = f[batch['cycle_life'][i, 0]].value
            policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
        # summary = {}
        # summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
        # summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
        # summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
        # summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
        # summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
        # summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
        # summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
        # summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
        # summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
        #     summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
        #            'cycle': summary_CY}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(cycles['I'].shape[0]):
                # I = np.hstack((f[cycles['I'][j, 0]].value))
                # Qc = np.hstack((f[cycles['Qc'][j, 0]].value))
                # Qd = np.hstack((f[cycles['Qd'][j, 0]].value))
                # Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]].value))
                # T = np.hstack((f[cycles['T'][j, 0]].value))
                # Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]].value))
                # V = np.hstack((f[cycles['V'][j, 0]].value))
                # dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]].value))
                # t = np.hstack((f[cycles['t'][j, 0]].value))
                I = f[cycles['I'][j, 0]].value
                Qc = f[cycles['Qc'][j, 0]].value
                Qd = f[cycles['Qd'][j, 0]].value
                Qdlin = f[cycles['Qdlin'][j, 0]].value
                T = f[cycles['T'][j, 0]].value
                Tdlin = f[cycles['Tdlin'][j, 0]].value
                V = f[cycles['V'][j, 0]].value
                dQdV = f[cycles['discharge_dQdV'][j, 0]].value
                t = f[cycles['t'][j, 0]].value
                cd = {'cycle_id': j, 'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                cycle_dict[j] = cd

            battery = {'cycle_life': cl, 'charge_policy': policy, 'cycles': cycle_dict}  # 'summary': summary,

            break
    return battery


