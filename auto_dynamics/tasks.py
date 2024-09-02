import seqio
import tensorflow as tf
import functools
from auto_dynamics import preprocessors
from typing import Mapping
import note_seq
import jax


# filename = '/home/grace/datasets/maestro/test/test.tfrecord'

# raw_dataset = tf.data.TFRecordDataset('/home/grace/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_train.tfrecord-00000-of-00025')
# for raw_record in raw_dataset.take(5):
#     with tf.io.TFRecordWriter(filename) as writer:
#         example = tf.train.Example()
#         example.ParseFromString(raw_record.numpy())
#         record_bytes = example.SerializeToString()
#         writer.write(record_bytes)

seqio.TaskRegistry.add(
    "midi_processing",
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            'train': '/home/grace/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_train.tfrecord-?????-of-00025',
            'eval': '/home/grace/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_validation.tfrecord-?????-of-00025'
        },
        feature_description={
            'sequence': tf.io.FixedLenFeature([], dtype=tf.string),
     }),
    output_features = {
            'inputs':
                seqio.Feature(seqio.PassThroughVocabulary(128), rank=2),
            'targets':
                seqio.Feature(seqio.PassThroughVocabulary(128))
            },
    preprocessors=[
        functools.partial(
            preprocessors.pass_through_processor,
            step_name='init'
        ),
        preprocessors.midi_processor,
        functools.partial(
            preprocessors.pass_through_processor,
            step_name='final'
        )
    ])

# dataset = seqio.get_mixture_or_task("remove_wav").get_dataset()
