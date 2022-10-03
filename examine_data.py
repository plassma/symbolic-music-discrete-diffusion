import numpy as np

if __name__ == '__main__':
    seq_len = 60 * 16

    midi_data = np.load('quantized_note_arrays.npy', allow_pickle=True)
    midi_data = [d[:seq_len].astype(int) for d in midi_data if len(d) >= seq_len]

    print(f'min: {min(midi_data)}, max: {max(midi_data)}')
