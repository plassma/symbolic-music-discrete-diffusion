import numpy as np
import matplotlib.pyplot as plt


def pit_range_hist(midi_data):
    midi_data = midi_data.copy()
    for i in range(len(midi_data)):
        midi_data[i][midi_data[i] == 0] = midi_data[i].max()

    ranges = midi_data.max(1) - midi_data.min(1)

    plt.hist(ranges)
    plt.show()

if __name__ == '__main__':
    midi_data = np.load('data/lakh_melody_64_1MIO.npy', allow_pickle=True)
    unique_pitches = np.unique(midi_data, axis=1)
    print(unique_pitches)