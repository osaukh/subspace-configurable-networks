import os
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.functional import resample
from typing import Optional, Tuple
from torch import Tensor

from typing import List
from tqdm import trange, tqdm
import math
import torch

HASH_DIVIDER = "_nohash_"

try:
    from torchaudio.datasets.speechcommands import load_speechcommands_item
except ImportError:
    def load_speechcommands_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        # Besides the officially supported split method for datasets defined by "validation_list.txt"
        # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
        # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
        # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
        # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
        # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
        # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
        speaker, _ = os.path.splitext(filename)
        speaker, _ = os.path.splitext(speaker)

        speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
        utterance_number = int(utterance_number)

        # Load audio
        waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate, label, speaker_id, utterance_number

try:
    from torchaudio.functional import speed
except ImportError:
    def speed(
        waveform: torch.Tensor, orig_freq: int, factor: float, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        source_sample_rate = int(factor * orig_freq)
        target_sample_rate = int(orig_freq)

        gcd = math.gcd(source_sample_rate, target_sample_rate)
        source_sample_rate = source_sample_rate // gcd
        target_sample_rate = target_sample_rate // gcd

        if lengths is None:
            out_lengths = None
        else:
            out_lengths = torch.ceil(lengths * target_sample_rate / source_sample_rate).to(lengths.dtype)            
        return resample(waveform, source_sample_rate, target_sample_rate), out_lengths


def transform_audio(transform_type, waveform, sample_rate, factor, device):
    allowed_audio_transformations = [
        "speed", "pitchshift", None
    ]

    if transform_type not in allowed_audio_transformations:
        raise NotImplementedError(f"{transform_type} has not been supported yet.")
    if transform_type =="pitchshift":
        waveform = torchaudio.functional.pitch_shift(waveform = waveform, sample_rate=sample_rate, n_steps=factor)
    if transform_type == "speed":
        assert factor<=1.0 and factor >=0, "No, you can only slow down the audio"
        new_waveform, _ = speed(waveform=waveform, orig_freq=sample_rate, factor=factor)
        empty_tensor = torch.ones([1,16000]).to(device)
        waveform_list = [empty_tensor.t()]
        for w in new_waveform:
            waveform_list.append(w.t())
        batch = torch.nn.utils.rnn.pad_sequence(waveform_list, batch_first=True, padding_value=0.)
        waveform = batch.permute(0, 2, 1)[1:]

    if transform_type is None:
        True
    return waveform.to(device)

class SpeechCommands(SPEECHCOMMANDS):
    # modified from: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
    def __init__(self, root:str="./data", subset: str = None, device=f'cuda:0', in_mem = True):
        super().__init__(root, download=True)
        self.device = device
        self.in_mem = in_mem
        self.labels = ['backward','bed','bird','cat','dog','down','eight','five','follow','forward','four','go','happy',
                       'house','learn','left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop',
                       'three','tree','two','up','visual','wow','yes','zero']
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

        self.id_of_speechchommands = torch.randint(len(self._walker), (len(self._walker), ))
        self.__waveforms = []
        self.__labels = []
        self.__fileid2id ={}
        if self.in_mem:
            self.__load_original_data_GPU__()
    
    def __load_original_data_GPU__(self):
        empty_tensor = torch.ones([1,16000])
        for i in tqdm(range(len(self._walker)), desc="Load original data to memory"):
            fileid = self._walker[i]
            waveform, sample_rate, label_word, speaker_id, utterance_number  = load_speechcommands_item(fileid, self._path)
            if waveform.shape[1]!=16000:
                batch = torch.nn.utils.rnn.pad_sequence([waveform.t(), empty_tensor.t()], batch_first=True, padding_value=0.)
                waveform = batch.permute(0, 2, 1)[0]
            self.__waveforms.append(waveform.to(self.device))    
            self.__labels.append(self.labels.index(label_word))
            self.__fileid2id[fileid]=i        

    def __getitem__(self, n: int) -> Tuple[Tensor, list[int]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, int):
            ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        if self.in_mem:
            result =  {}
            id = self.id_of_speechchommands[n]
            fileid = self._walker[id]
            result[f'label'] = self.__labels[self.__fileid2id[fileid]]
            result[f'waveform'] = self.__waveforms[self.__fileid2id[fileid]]
            return result

        # else:
        result = {}
        empty_tensor = torch.ones([1,16000])
        id = self.id_of_speechchommands[n]
        fileid = self._walker[id]
        waveform, sample_rate, label_word, speaker_id, utterance_number  = load_speechcommands_item(fileid, self._path)
        label = self.labels.index(label_word)

        
        if waveform.shape[1]!=16000:
            batch = torch.nn.utils.rnn.pad_sequence([waveform.t(), empty_tensor.t()], batch_first=True, padding_value=0.)
            waveform = batch.permute(0, 2, 1)[0]
                        
        result[f'waveform'] = waveform
        result[f'label'] = torch.tensor(label)


        return result
    
    def label_to_index(self,word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))


    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]

