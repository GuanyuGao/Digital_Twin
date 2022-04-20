import h5py
import numpy as np

batch_1 = './data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
batch_2 = './data/2017-06-30_batchdata_updated_struct_errorcorrect.mat'
batch_3 = './data/2018-04-12_batchdata_updated_struct_errorcorrect.mat'


def all_batteries():

    batteries_1 = get_batteries(batch_1)
    batteries_2 = get_batteries(batch_2)
    batteries_3 = get_batteries(batch_3)
    batteries = {}
    for key, value in batteries_1:
        batteries[key] = value
    for key, value in batteries_2:
        batteries[key] = value
    for key, value in batteries_3:
        batteries[key] = value
    return batteries


def get_batteries(matFilename):

    f = h5py.File(matFilename)
    batch = f['batch']
    num_cells = batch['summary'].shape[0]
    batteries = {}
    cell_dict = {}

    for i in range(num_cells):

        cl = f[batch['cycle_life'][i, 0]].value
        policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
        barcode = f[batch['barcode'][i, 0]].value.tobytes()[::2].decode()

        summary = {}
        summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
        summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
        summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
        summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
        summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
        summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
        summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
        summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
            summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                   'cycle': summary_CY}
        cycles = f[batch['cycles'][i, 0]]
        cycle_dict = {}
        for j in range(cycles['I'].shape[0]):
            I = np.hstack((f[cycles['I'][j, 0]].value))
            Qc = np.hstack((f[cycles['Qc'][j, 0]].value))
            Qd = np.hstack((f[cycles['Qd'][j, 0]].value))
            Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]].value))
            T = np.hstack((f[cycles['T'][j, 0]].value))
            Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]].value))
            V = np.hstack((f[cycles['V'][j, 0]].value))
            dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]].value))
            t = np.hstack((f[cycles['t'][j, 0]].value))
            cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
            cycle_dict[str(j)] = cd

        cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
        batteries = {barcode: cell_dict}
    return batteries


