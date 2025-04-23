import wfdb
import numpy as np

def load_record(record_name, pn_dir='mitdb', window_size=100):
    """
    Načíta signál a anotácie zo záznamu MIT-BIH a vráti segmentované beaty a labely.

    Args:
        record_name (str): názov záznamu, napr. '100'
        pn_dir (str): názov PhysioNet databázy (default = 'mitdb')
        window_size (int): počet vzoriek pred a po R-vlne

    Returns:
        beats (np.ndarray): shape [n_beats, 2*window_size]
        labels (np.ndarray): shape [n_beats], 0 = normálne, 1 = abnormálne
    """
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    annotation = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)

    signal = record.p_signal[:, 0]  # len 1. kanál
    beats = []
    labels = []

    for i in range(len(annotation.sample)):
        r = annotation.sample[i]
        if r - window_size < 0 or r + window_size > len(signal):
            continue  # vynechaj okraj

        segment = signal[r - window_size : r + window_size]
        label = annotation.symbol[i]

        # Normálne = 'N', iné = abnormálne
        binary_label = 0 if label == 'N' else 1
        beats.append(segment)
        labels.append(binary_label)

    return np.array(beats), np.array(labels)
