from pathlib import Path


class Paths:
    
    def __init__(self, data_dir: str, checkpoint_dir: str, dataset_dir: str, precomputed_mels: str, metadata_path: str, wav_path: str, text_path:str,wav_test_path:str, text_test_path:str):
        self.data_dir = Path(data_dir)
        self.dataset_dir = dataset_dir
        self.wav_path = wav_path
        self.text_path = text_path
        self.metadata_path = Path(metadata_path)
        self.mel_dir = self.data_dir / 'mels'
        self.token_dir = self.data_dir / 'tokens'
        self.precomputed_mels = precomputed_mels
        if self.precomputed_mels is not None:
            self.precomputed_mels = Path(precomputed_mels)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.text_test_path =text_test_path
        self.wav_test_path=wav_test_path
        self.create_dirs()
    
    def create_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mel_dir.mkdir(parents=True, exist_ok=True)
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config(cls, config):
        return cls(
            data_dir=config['data_dir'],
            metadata_path=config['metadata_path'],
            checkpoint_dir=config['checkpoint_dir'],
            dataset_dir=config['dataset_dir'],
            precomputed_mels=config['precomputed_mels'],
            wav_path=config['wav_path'],
            text_path=config['text_path'],
            text_test_path=config['text_test_path'],
            wav_test_path=config["wav_test_path"],
            
        )
