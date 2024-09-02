import note_seq
from absl import app



def main(args):
    midi_file_name_1 = '/home/grace/ch.midi'
    midi_file_name_2 = '/home/grace/transcribed_human_chopin.mid'

    midi_note_seq_1 = note_seq.midi_file_to_note_sequence(midi_file_name_1)
    midi_note_seq_2 = note_seq.midi_file_to_note_sequence(midi_file_name_2)

    sum_squared = 0
    for i, note in enumerate(midi_note_seq_1.notes):
        velocity_1 = note.velocity
        velocity_2 = midi_note_seq_2.notes[i].velocity
        sum_squared+=(velocity_1-velocity_2)**2
    print(sum_squared/len(midi_note_seq_1.notes))

        

if __name__ == '__main__':
    app.run(main=main)
