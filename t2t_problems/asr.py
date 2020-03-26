import os
import numpy as np

from tensor2tensor.data_generators import audio_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_audio
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics

import tensorflow as tf


class SubwordTextEncoderWithEos(text_encoder.SubwordTextEncoder):
    """Encodes each byte to an id and appends the EOS token."""

    def encode(self, s):
        return super(SubwordTextEncoderWithEos, self).encode(s) + [text_encoder.EOS_ID]


class AsrProblem(problem.Problem):
    """Base class for speech recognition problems."""

    def hparams(self, defaults, model_hparams):
        params = model_hparams
        # Filterbank extraction in bottom instead of preprocess_example is faster.
        params.add_hparam("audio_preproc_in_bottom", False)
        # The trainer seems to reserve memory for all members of the input dict
        params.add_hparam("audio_keep_example_waveforms", False)
        if self.is_8k:
            params.add_hparam("audio_sample_rate", 8000)
            params.add_hparam("audio_preemphasis", 0.97)
            params.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
            params.add_hparam("audio_frame_length", 25.0)
            params.add_hparam("audio_frame_step", 10.0)
            params.add_hparam("audio_lower_edge_hertz", 20.0)
            params.add_hparam("audio_upper_edge_hertz", 4000.0)
            params.add_hparam("audio_num_mel_bins", 40)
            params.add_hparam("audio_add_delta_deltas", True)
            params.add_hparam("num_zeropad_frames", 250)
        else:
            params.add_hparam("audio_sample_rate", 16000)
            params.add_hparam("audio_preemphasis", 0.97)
            params.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
            params.add_hparam("audio_frame_length", 25.0)
            params.add_hparam("audio_frame_step", 10.0)
            params.add_hparam("audio_lower_edge_hertz", 20.0)
            params.add_hparam("audio_upper_edge_hertz", 8000.0)
            params.add_hparam("audio_num_mel_bins", 80)
            params.add_hparam("audio_add_delta_deltas", True)
            params.add_hparam("num_zeropad_frames", 250)

        params = defaults
        params.modality = {
            "inputs": modalities.ModalityType.SPEECH_RECOGNITION,
            "targets": modalities.ModalityType.SYMBOL
        }
        params.vocab_size = {
            "inputs": None,
            "targets": self.get_feature_encoders()["targets"].vocab_size
        }

    @property
    def approx_vocab_size(self):
        raise NotImplementedError()

    @property
    def input_space_id(self):
        return problem.SpaceID.AUDIO_SPECTRAL

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_BPE_TOK

    def feature_encoders(self, data_dir):
        if self.is_8k:
            sample_rate = 8000
        else:
            sample_rate = 16000
        encoders = {
            "inputs": None,
            "waveforms": audio_encoder.AudioEncoder(sample_rate=sample_rate)
        }
        vocab_file = os.path.join(data_dir, "vocab", "bpe.%d.t2t" % self.approx_vocab_size)
        target_encoder = SubwordTextEncoderWithEos(filename=vocab_file)
        encoders["targets"] = target_encoder
        return encoders

    def example_reading_spec(self):
        data_fields = {
            "waveforms": tf.VarLenFeature(tf.float32),
            "targets": tf.VarLenFeature(tf.int64),
        }

        data_items_to_decoders = None

        return data_fields, data_items_to_decoders

    def preprocess_example(self, example, mode, hparams):
        params = hparams
        if params.audio_preproc_in_bottom:
            example["inputs"] = tf.expand_dims(
                tf.expand_dims(example["waveforms"], -1), -1)
        else:
            waveforms = tf.expand_dims(example["waveforms"], 0)
            mel_fbanks = common_audio.compute_mel_filterbank_features(
                waveforms,
                sample_rate=params.audio_sample_rate,
                dither=params.audio_dither,
                preemphasis=params.audio_preemphasis,
                frame_length=params.audio_frame_length,
                frame_step=params.audio_frame_step,
                lower_edge_hertz=params.audio_lower_edge_hertz,
                upper_edge_hertz=params.audio_upper_edge_hertz,
                num_mel_bins=params.audio_num_mel_bins,
                apply_mask=False)
            if params.audio_add_delta_deltas:
                mel_fbanks = common_audio.add_delta_deltas(mel_fbanks)
            fbank_size = common_layers.shape_list(mel_fbanks)
            assert fbank_size[0] == 1

            # This replaces CMVN estimation on data
            var_epsilon = 1e-09
            mean = tf.reduce_mean(mel_fbanks, keepdims=True, axis=1)
            variance = tf.reduce_mean(tf.squared_difference(mel_fbanks, mean),
                                      keepdims=True, axis=1)
            mel_fbanks = (mel_fbanks - mean) * tf.rsqrt(variance + var_epsilon)

            # Later models like to flatten the two spatial dims. Instead, we add a
            # unit spatial dim and flatten the frequencies and channels.
            example["inputs"] = tf.concat([
                tf.reshape(mel_fbanks, [fbank_size[1],
                                        fbank_size[2], fbank_size[3]]),
                tf.zeros((params.num_zeropad_frames, fbank_size[2], fbank_size[3]))], 0)

        if not params.audio_keep_example_waveforms:
            del example["waveforms"]
        return super(AsrProblem, self).preprocess_example(example, mode, hparams)

    def eval_metrics(self):
        defaults = super(AsrProblem, self).eval_metrics()
        defaults.append(metrics.Metrics.EDIT_DISTANCE)
        return defaults
