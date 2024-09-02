import uuid
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import os

import functools
import gin
import jax
import librosa
import note_seq
import seqio
import t5
import t5x
from t5x import partitioning as t5x_partitioning
from auto_dynamics.example import network
import note_seq
from auto_dynamics import preprocessors
from auto_dynamics import models


SAMPLE_RATE = 16000
FLAGS = flags.FLAGS
EOS_ID = -1


class InferenceModel(object):
  """Wrapper of T5X model for music transcription."""

  def __init__(self, checkpoint_path):
    self.inputs_length = 256

    gin_files = ['auto_dynamics/example/infer.gin']

    self.batch_size = 1
    self.outputs_length = 2048
    self.sequence_length = {'inputs': self.inputs_length,
                            'targets': self.outputs_length}

    self.partitioner = t5x_partitioning.PjitPartitioner(
        num_partitions=1)

    self.output_features = {
        'inputs': seqio.Feature(seqio.PassThroughVocabulary(128), rank=2),
        'targets': seqio.Feature(seqio.PassThroughVocabulary(128))
    }

    # Create a T5X model.
    self._parse_gin(gin_files)
    self.model = self._load_model()

    # Restore from checkpoint.
    self.restore_from_checkpoint(checkpoint_path)

  @property
  def input_shapes(self):
    return {
        'encoder_input_tokens': (self.batch_size, self.inputs_length, 88),
        'decoder_input_tokens': (self.batch_size, self.outputs_length)
    }

  def _parse_gin(self, gin_files):
    """Parse gin files used to train the model."""
    gin_bindings = [
        'from __gin__ import dynamic_registration'
    ]
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          gin_files, gin_bindings, finalize_config=False)

  def _load_model(self):
    """Load up a T5X `Model` after parsing training gin config."""
    model_config = gin.get_configurable(network.T5Config)()
    module = network.Transformer(config=model_config)
    return models.CustomizedEncoderDecoderModel(
        module=module,
        input_vocabulary=self.output_features['inputs'].vocabulary,
        output_vocabulary=self.output_features['targets'].vocabulary,
        optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0))

  def restore_from_checkpoint(self, checkpoint_path):
    """Restore training state from checkpoint, resets self._predict_fn()."""
    train_state_initializer = t5x.utils.TrainStateInitializer(
        optimizer_def=self.model.optimizer_def,
        init_fn=self.model.get_initial_variables,
        input_shapes=self.input_shapes,
        partitioner=self.partitioner)

    restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
        path=checkpoint_path, mode='specific', dtype='float16')

    train_state_axes = train_state_initializer.train_state_axes
    self._predict_fn = self._get_predict_fn(train_state_axes)
    self._train_state = train_state_initializer.from_checkpoint_or_scratch(
        [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))

  @functools.lru_cache()
  def _get_predict_fn(self, train_state_axes):
    """Generate a partitioned prediction function for decoding."""

    def partial_predict_fn(params, batch, decode_rng):
      return self.model.predict_batch_with_aux(
          params, batch, decoder_params={'decode_rng': None})

    return self.partitioner.partition(
        partial_predict_fn,
        in_axis_resources=(
            train_state_axes.params,
            t5x.partitioning.PartitionSpec('data', ), None),
        out_axis_resources=t5x.partitioning.PartitionSpec('data', )
    )

  def predict_tokens(self, batch, seed=0):
    """Predict tokens from preprocessed dataset batch."""
    prediction, _ = self._predict_fn(
        self._train_state.params, batch, jax.random.PRNGKey(seed))
    
    return prediction

  def __call__(self, midi_file):
    
    ds, original_input = self.midi_file_to_dataset(midi_file=midi_file)
    # ds = self.preprocess(ds)

    model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
        ds, task_feature_lengths=self.sequence_length)
    model_ds = model_ds.batch(self.batch_size)

    # inferences = (tokens for batch in model_ds.as_numpy_iterator()
    #     for tokens in self.predict_tokens(batch))

    target_lengths = []
    orig_predictions = []
    for batch in model_ds.as_numpy_iterator():
        target_lengths.append(list(batch['decoder_target_tokens'][0]).index(-1))
        for tokens in self.predict_tokens(batch):
          orig_predictions.append(tokens)

    
    nns = note_seq.NoteSequence()
    
    pitch_predictions = []
    velocity_predictions = []
    duration_predictions = []
    for j, arr in enumerate(orig_predictions):
      for i, num in enumerate(arr):
        if (target_lengths[j] <= i):
          break
        if (i % 3 == 0):
          pitch_predictions.append(int(num))
        elif (i % 3 == 1): duration_predictions.append(int(num))
        else: velocity_predictions.append(int(num))
             
    for i, arr in enumerate(original_input):
        n = nns.notes.add()
        
        n.pitch = int(original_input[i][0])
        n.start_time = original_input[i][1]
        n.end_time = original_input[i][2]
        n.velocity = velocity_predictions[i]
        
    print(nns.notes[0])

    # for i, arr in enumerate(predictions):
    #   for j, velocity in enumerate(arr):
    #     original_input_index = j + i*self.inputs_length

    #     n = nns.notes.add()
    #     n.pitch = original_input[original_input_index][0]
    #     n.start_time = original_input[original_input_index][1]
    #     n.end_time = original_input[original_input_index][2]
    #     n.velocity = velocity
    

    return nns

  def midi_file_to_dataset(self, midi_file):

    length = 256
    target_length = 2048 

    def get_data(ex):
        ns = note_seq.NoteSequence.FromString(ex)
        notes = sorted(ns.notes,
                 key=lambda note: (round(100*note.start_time), note.pitch))

        smallest_difference = 100
        previous_start_time = 0
        for note in notes:
          start_time = round(100*note.start_time) #time in units of 10ms
          if (start_time - previous_start_time > 0):
            smallest_difference = min(smallest_difference, start_time-previous_start_time)
          previous_start_time = start_time


        orig_input = np.zeros((length, 88)).astype(int)
        decoder_targets = np.zeros((target_length)).astype(float)

        decoder_length_index = 0
        subtract_time = 0
        TIME_SCALE = 100/smallest_difference
        TIME_SCALE=100
        tf.print("notes len: {}", len(notes))
        processed_input = 0
        produced_output = 0
        for note in notes:
          processed_input += 1
          start_time = round(TIME_SCALE*note.start_time) 
          end_time = round(TIME_SCALE*note.end_time)
          duration = round(TIME_SCALE*note.end_time - TIME_SCALE*note.start_time)
          pitch_index = note.pitch - 21 # 21-109
          velocity = note.velocity 

          while (start_time - subtract_time >= length):
            decoder_targets[decoder_length_index*3] = -1
            decoder_targets = decoder_targets[:decoder_length_index*3+1]
            if (decoder_length_index != 0):
              produced_output += (len(decoder_targets) - 1) / 2
              yield {
              'inputs': orig_input,
              'targets': decoder_targets
              }
            

            orig_input = np.zeros((length, 88)).astype(int)
            decoder_targets = np.zeros((target_length)).astype(float)
            decoder_length_index = 0
            subtract_time += length
            
          
          if (start_time==end_time or duration == 0):
            orig_input[start_time - subtract_time][pitch_index] = 2 # note articulated
          else:
            curr_time = start_time
            while (curr_time < end_time and curr_time - subtract_time < length):
              orig_input[curr_time - subtract_time][pitch_index] = 1
              curr_time+=1
            if (end_time<length):
              orig_input[end_time - subtract_time][pitch_index] = 0
          
          # decoder_targets[decoder_length_index] = velocity/127
          decoder_targets[decoder_length_index*3] = pitch_index 
          decoder_targets[decoder_length_index*3+1] = velocity
          decoder_targets[decoder_length_index*3+2] = duration
          # decoder_targets[decoder_length_index*2+1] = 0 
          decoder_length_index+=1
          
        decoder_targets[decoder_length_index*3] = -1
        decoder_targets = decoder_targets[:decoder_length_index*3+1]
        if (decoder_length_index != 0):
          yield {
              'inputs': orig_input,
              'targets': decoder_targets
              }

    def process_example(ex):
      example_ds = tf.data.Dataset.from_generator(
        get_data,
        output_signature={
            'inputs':
                tf.TensorSpec(shape=(self.inputs_length, 88), dtype=tf.int32),
            'targets':
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
        },
        args=[ex])
      return example_ds
    
    def get_input():
      ns = note_seq.NoteSequence.FromString(ex)
      notes = sorted(ns.notes,
                 key=lambda note: (round(100*note.start_time), note.pitch))

      orig_input = np.zeros((len(notes), 3)).astype(float)
      length_index = 0
      for note in notes:
          start_time = note.start_time #time in units of 10ms
          end_time = note.end_time
          pitch = note.pitch

          orig_input[length_index][0] = pitch
          orig_input[length_index][1] = start_time
          orig_input[length_index][2] = end_time

          length_index +=1
      return orig_input
    
      
    
    midi_note_seq = note_seq.midi_file_to_note_sequence(midi_file)
    ex = midi_note_seq.SerializeToString()

    return process_example(ex), get_input()

  def preprocess(self, ds):
    pp_chain = [
        # functools.partial(
        #     t5.data.preprocessors.split_tokens_to_inputs_length,
        #     sequence_length=self.sequence_length,
        #     output_features=self.output_features,
        #     feature_key='inputs',
        #     additional_feature_keys = ['targets'])
    ]
    for pp in pp_chain:
      ds = pp(ds)
    return ds


def get_transcription_b64(model: InferenceModel, midi_file) -> str:
  est_ns = model(midi_file)
  tmp_path = gen_tmp_path()
  note_seq.sequence_proto_to_midi_file(est_ns, tmp_path)
  try:
    with open(tmp_path, "rb") as f:
      data = f.read()
    remove_file_if_exists(tmp_path)
    return data
  except:
    remove_file_if_exists(tmp_path)
    raise NotImplementedError()


def remove_file_if_exists(path: str):
  if os.path.exists(path):
    os.remove(path)


def gen_tmp_path():
  return f"/tmp/{uuid.uuid4()}.midi"




def main(args):
  app.parse_flags_with_usage(args)
  midi_file = '/home/grace/chopin_op25_e1.mid'
  piano_model = InferenceModel('/home/grace/model/mse1hot/checkpoint_3000000')

  midi_bin = get_transcription_b64(piano_model, midi_file)
  with open('/home/grace/my_model_chopin.midi', 'wb') as m:
    m.write(midi_bin)


if __name__ == '__main__':
  app.run(main=main)
