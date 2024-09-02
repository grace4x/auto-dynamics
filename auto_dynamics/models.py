from typing import Mapping, Optional, Tuple, Any
import seqio
from t5x import decoding
from t5x import models
import t5x
from flax import linen as nn
import tensorflow as tf
import jax.numpy as jnp
import jax
from t5x import metrics as metrics_lib
from flax.training import common_utils
import math

MetricsMap = metrics_lib.MetricsMap
PyTree = Any
PyTreeDef = jax.tree_util.PyTreeDef


class CustomizedFeatureConverter(seqio.FeatureConverter):
  """Feature converter for an encoder-decoder architecture.

  The input dataset has "inputs" and "targets" field. These will be converted
  to a subset of standard features.

  To use packing, pass pack = True argument to the FeatureConverter's
  constructor. When packing is done, two additional fields are added for each of
  "inputs" and "targets" fields.

  Example for a packed dataset:

  The input dataset has two examples each with "inputs" and "targets".

  ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

  task_feature_lengths = {"inputs": 10, "targets": 7}

  First, the `inputs` are packed together, padded to length 10 and assigned to
  "encoder_input_tokens" field. The `targets` are processed similarly.

  The "*_segment_id" fields are generated from the packing operation. For the
  explanation of these fields, see the module docstring.

  The "decoder_loss_weights" is a binary mask indicating where non-padding
  positions are, i.e., value of 1 indicates non-padding and 0 for padding. This
  class assumes that the loss is taken only on the decoder side.

  converted_ds = [{
       "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
          "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
      "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
       "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
       "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
          "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
  }]

  Note that two examples are packed together into one example.
  """

  TASK_FEATURES = {
      "inputs": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32, rank=2),
      # "targets": seqio.FeatureConverter.FeatureSpec(dtype=tf.float32),
      "targets": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32, rank=2),
      # "decoder_target_tokens": seqio.FeatureConverter.FeatureSpec(dtype=tf.float32),
      "decoder_target_tokens": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      # "decoder_input_tokens": seqio.FeatureConverter.FeatureSpec(dtype=tf.float32),
      "decoder_input_tokens": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "encoder_segment_ids": tf.int32,
      "decoder_segment_ids": tf.int32,
      "encoder_positions": tf.int32,
      "decoder_positions": tf.int32,
  }

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Convert a seq2seq example into an example with model features."""
    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None),
        bos_id=self.bos_id,
    )

    d = {
        "encoder_input_tokens": features["inputs"],
        "decoder_target_tokens": features["targets"],
        "decoder_input_tokens": decoder_input_tokens,
        # Loss is computed for all but the padding positions.
        "decoder_loss_weights": seqio.non_padding_position(features["targets"]),
    }
    d.update({k: features[k] for k in self._passthrough_features})

    if self.pack:
      d["encoder_segment_ids"] = features["inputs_segment_ids"]
      d["decoder_segment_ids"] = features["targets_segment_ids"]
      d["encoder_positions"] = features["inputs_positions"]
      d["decoder_positions"] = features["targets_positions"]

    return d

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    The conversion process involves two steps

    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.

    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """
    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = task_feature_lengths["targets"]

    model_feature_lengths = {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
    }
    for k in self._passthrough_features:
      model_feature_lengths[k] = task_feature_lengths[k]

    if self.pack:
      model_feature_lengths["encoder_segment_ids"] = encoder_length
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["encoder_positions"] = encoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths

class CustomizedEncoderDecoderModel(models.EncoderDecoderModel):

  FEATURE_CONVERTER_CLS = CustomizedFeatureConverter
      
  def __init__(self, module, input_vocabulary, output_vocabulary, optimizer_def,
               loss_double_mse=False, loss_3hot=False, decode_fn=decoding.beam_search, label_smoothing=0.0,
               z_loss=0.0, loss_normalizing_factor=None):
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor)
    self._loss_double_mse = loss_double_mse
    self._loss_3hot = loss_3hot

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array],
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Loss function used for training with a MSE loss."""

    logits = self._compute_logits(params, batch, dropout_rng)

    loss_normalizing_factor: Optional[
        Union[float, int, str, losses.SpecialLossNormalizingFactor]
    ]

    targets = batch['decoder_target_tokens']

    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            'Incorrect shapes. Got shape %s logits and %s targets'
            % (str(logits.shape), str(targets.shape))
        )
    

    soft_targets = common_utils.onehot(
      targets, num_classes=128, on_value=1.0, off_value=0.0)

    if self._loss_3hot:
      targets_left = common_utils.onehot(
        targets-1, num_classes=128, on_value=0.5, off_value=0.0
      )
      targets_right = common_utils.onehot(
        targets+1, num_classes=128, on_value=0.5, off_value=0.0
      )
      soft_targets += targets_left + targets_right

    logits_softmax = jax.nn.softmax(logits, axis=-1)
    weights = batch.get('decoder_loss_weights', None)
    weight_sum = jnp.sum(weights)
    token_loss = jnp.mean(jnp.square(soft_targets - logits_softmax), axis=-1)

    if self._loss_double_mse:
      loss = jnp.sum(jnp.square(token_loss * weights)) / weight_sum
    else:
      loss = jnp.sum(token_loss * weights) / weight_sum

    # logits_argmax = jnp.argmax(logits, axis=-1)
    # weights = batch.get('decoder_loss_weights', None)
    # total_loss = jnp.mean(weights * jnp.square(targets - logits_argmax), -1)
    # loss = sum(total_loss)/100000
    
    # segment ids to compute packing, padding etc.
    segment_ids = {
        k[: -len('_segment_ids')]: v
        for k, v in batch.items()
        if k.endswith('_segment_ids')
    }
    # If these don't exist then we can create only padding mask.
    if not segment_ids:
      segment_ids = {
          k: v != 0
          for k, v in batch.items()
          if k in ('encoder_input_tokens', 'decoder_target_tokens')
      }
    
    metrics = self._compute_metrics(
        logits=logits,
        targets=targets,
        mask=weights,
        loss=loss    
        )
    return loss, metrics
