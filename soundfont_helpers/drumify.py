import note_seq
import sys

if __name__ == '__main__':
    ns = note_seq.midi_file_to_note_sequence(sys.argv[1])
    for n in ns.notes:
        n.is_drum = True
        n.instrument = 2
    note_seq.note_sequence_to_midi_file(ns, sys.argv[1])
