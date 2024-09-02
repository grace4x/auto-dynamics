import tensorflow as tf
from typing import Mapping, Any
import note_seq
import numpy as np
import random

length = 256
target_length = 1024
EOS = -1
SMALLEST_UNIT_IN_SECONDS = 0.05
SCALE_FACTOR = 1/SMALLEST_UNIT_IN_SECONDS
# NUM_SAMPLES = 200

def midi_processor(
    ds: tf.data.Dataset
) -> tf.data.Dataset:

    def get_data(sequence):
      ns = note_seq.NoteSequence.FromString(sequence)
      notes = sorted(ns.notes,
                key=lambda note: (round(SCALE_FACTOR*note.start_time), note.pitch))
      
      NUM_SAMPLES = len(notes)//5

      # smallest_difference = 1000
      # previous_start_time = 0
      # for note in notes:
      #   start_time = round(SCALE_FACTOR*note.start_time) #time in units of 50ms
      #   if (start_time - previous_start_time > 0):
      #     smallest_difference = min(smallest_difference, start_time-previous_start_time)
      #   previous_start_time = start_time

      # TIME_SCALE = SCALE_FACTOR/smallest_difference
      TIME_SCALE = SCALE_FACTOR


      for i in range(NUM_SAMPLES):
        
        start_index = random.randint(0, len(notes)-1)

        encoder_input = np.zeros((length, 88)).astype(int)
        decoder_targets = np.zeros((target_length)).astype(float)
        decoder_length_index = 0
        subtract_time = round(TIME_SCALE*notes[start_index].start_time)

        for i in range(start_index, len(notes)):

          note = notes[i]

          start_time = round(TIME_SCALE*note.start_time) 
          end_time = round(TIME_SCALE*note.end_time)
          duration = round(TIME_SCALE*note.end_time - TIME_SCALE*note.start_time)
          pitch_index = note.pitch - 21  # range 21-109
          velocity = note.velocity 

          if (start_time - subtract_time >= length or decoder_length_index*3 >= target_length - 1):
            
            decoder_targets[decoder_length_index*3] = -1
            decoder_targets = decoder_targets[:decoder_length_index*3+1]
            
            if (decoder_length_index != 0):
              yield {
              'inputs': encoder_input,
              'targets': decoder_targets
              }

            break
          
          curr_time = start_time
          if (start_time==end_time or duration == 0):
            encoder_input[start_time - subtract_time][pitch_index] = 2  # note articulated
            curr_time+=1
          else:
            while (curr_time < end_time and curr_time - subtract_time < length):
              encoder_input[curr_time - subtract_time][pitch_index] = 1
              curr_time+=1

          decoder_targets[decoder_length_index*3] = pitch_index 
          decoder_targets[decoder_length_index*3+1] = curr_time - start_time # duration
          decoder_targets[decoder_length_index*3+2] = velocity
          decoder_length_index+=1


    def process_example(example):
      new_ds = tf.data.Dataset.from_generator(
        get_data,
        output_signature={
            'inputs':
                tf.TensorSpec(shape=(length, 88), dtype=tf.int32),
            'targets':
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
        },
        args=[example['sequence']])
      return new_ds
    
    return ds.flat_map(process_example)

def pass_through_processor(
    ds: tf.data.Dataset,
    step_name: str
) -> tf.data.Dataset:
  def _print_example(ex):
    filename = '/home/grace/datasets_cache/debugger/' + step_name + ".tfrecord"
    for key, value in ex.items():
      ts = tf.convert_to_tensor(value)
      if key == "audio":
        tf.print(key, str(ts.dtype), ts.shape, tf.size(ts),  "<...wav>", output_stream="file://" + filename)
      elif key == "sequence":
        tf.print(key, str(ts.dtype), ts.shape, "<...midi sequence>", output_stream="file://" + filename)
      else:
        tf.print(key,str(ts.dtype), ts.shape, tf.size(ts), ts, output_stream="file://" + filename, summarize=-1)
    tf.print("\n", output_stream="file://" + filename)
    return ex
  return ds.map(_print_example,
                num_parallel_calls=1,
                deterministic=True)

