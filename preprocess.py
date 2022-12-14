import argparse
import json
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, Union
import whisper
import numpy as np
import tqdm

from dfa.audio import Audio
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import get_files, read_config, pickle_binary, read_metafile, read_lyric


class Preprocessor:
    """Performs mel extraction and tokenization and stores the resulting torch tensors."""
    
    def __init__(self,
                 audio: Audio,
                 tokenizer: Tokenizer,
                 paths: Paths,
                 text_dict: Dict[str, str],
                 mel_dim_last=True) -> None:
        self.audio = audio
        self.paths = paths
        self.tokenizer = tokenizer
        self.text_dict = text_dict
        self.mel_dim_last = mel_dim_last
        self.model = whisper.load_model("tiny")
        
    def __call__(self, file_path: Path) -> Dict[str, Union[str, int]]:
        item_id = file_path.stem
        if self.paths.precomputed_mels:
            mel = np.load(self.paths.precomputed_mels / f'{item_id}.npy')
            if mel.shape[1] != self.audio.n_mels:
                mel = mel.T
            assert mel.shape[1] == self.audio.n_mels, \
                f'Expected mel shape to be of (None, {self.audio.n_mels}), but was: {mel.shape}! ' \
                f'Consider setting config/audio/mel_dim_last: {not self.mel_dim_last}'
        else:
            # wav = self.audio.load_wav(file_path)
            # mel = self.audio.wav_to_mel(wav)
            audio = whisper.load_audio(file_path)
            l =int(len(audio)/320)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(self.model.device)
            mel = self.model.embed_audio(mel).cpu().detach().numpy()[0][:l,]
            assert mel.shape[1] == self.audio.n_mels, \
                    f'Expected mel shape to be of (None, {self.audio.n_mels}), but was: {mel.shape}! ' \
                    f'Consider setting config/audio/mel_dim_last: {not self.mel_dim_last}'

        
        np.save(self.paths.mel_dir / f'{item_id}.npy', mel, allow_pickle=False)
        text = self.text_dict[item_id]
        tokens = np.array(self.tokenizer(text)).astype(np.int32)
        # print("token:",len(tokens))
        np.save(self.paths.token_dir / f'{item_id}.npy', tokens, allow_pickle=False)
        return {'item_id': item_id, 'tokens_len': tokens.shape[0], 'mel_len': mel.shape[0]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', help='Points to the config file.', default='config.yaml')
    parser.add_argument('--num_workers', '-w', metavar='N', type=int, default=cpu_count() - 1,
                        help='The number of worker threads to use for preprocessing')

    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths.from_config(config['paths'])
    audio = Audio.from_config(config['audio'])
    
    mel_dim_last = config['preprocessing']['mel_dim_last']
    
    print(f'Config: {args.config}\n'
          f'Target data directory: {paths.data_dir}')
    
    text_dict = read_lyric(paths.text_path, paths.text_test_path)
    # print(text_dict)
    with open("text_dict.json",'w') as f:
        json.dump(text_dict,f,ensure_ascii=False,indent=4)
    # text_dict = read_metafile(paths.metadata_path)
    
    symbols = set()
    for text in text_dict.values():
        symbols.update(set(text))
    symbols = sorted(list(symbols))

    if paths.precomputed_mels:
        audio_files = get_files(paths.precomputed_mels, extension='.npy')
    else:
        audio_files = get_files(paths.wav_path, extension='.wav')
        audio_files.extend(get_files(paths.wav_test_path, extension='.wav'))
        
    audio_files = [x for x in audio_files if x.stem in text_dict]
    tokenizer = Tokenizer(symbols)
    preprocessor = Preprocessor(audio=audio, tokenizer=tokenizer, paths=paths,
                                text_dict=text_dict, mel_dim_last=mel_dim_last)
    # pool = Pool(processes=args.num_workers)
    # mapper = pool.imap_unordered(preprocessor, audio_files)
    dataset = []
    # for i, item in tqdm.tqdm(enumerate(mapper), total=len(audio_files)):
    for i, item in tqdm.tqdm(enumerate(audio_files), total=len(audio_files)):
        dataset.append(preprocessor(item))
    
    pickle_binary(dataset, paths.data_dir / 'dataset.pkl')
    pickle_binary(symbols, paths.data_dir / 'symbols.pkl')
    print('Preprocessing done.')
