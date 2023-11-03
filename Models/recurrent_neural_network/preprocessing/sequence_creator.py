import numpy as np


def create_sequences(data, labels, sequence_length):
    """
    Create sequences from the provided features and labels.
    """
    sequences = []
    sequence_labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i : i + sequence_length])
        sequence_labels.append(labels[i + sequence_length])
    return np.array(sequences), np.array(sequence_labels)
