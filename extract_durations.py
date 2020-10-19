from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import tqdm
import argparse
from multiprocessing.pool import Pool

from dfa.audio import Audio
from dfa.dataset import new_dataloader
from dfa.duration_extraction import extract_durations_with_dijkstra
from dfa.model import Aligner
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import read_config, to_device, unpickle_binary, get_files


def extract_durations_for_item(item: dict, token_file: Path, pred_file: Path) -> Tuple[dict, np.array]:
    tokens_len, mel_len = item['tokens_len'], item['mel_len']
    tokens = np.load(str(token_file), allow_pickle=False)
    tokens = tokens[:tokens_len]
    pred = np.load(str(pred_file), allow_pickle=False)
    pred = pred[:, :mel_len, :]
    durations = extract_durations_with_dijkstra(tokens, pred)
    return item, durations


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', required=True, type=str, help='Points to the config file.')
    parser.add_argument('--model', '-m', default=None, type=str, help='Points to the a model file to restore.')
    parser.add_argument('--target', '-t', default='output', type=str, help='Target path')
    parser.add_argument('--batch_size', '-b', default=8, type=int, help='Batch size for inference.')
    parser.add_argument('--num_workers', '-w', metavar='N', type=int, default=cpu_count() - 1,
                        help='The number of worker threads to use for preprocessing')

    args = parser.parse_args()
    config = read_config(args.config)
    paths = Paths(**config['paths'])
    model_path = args.model if args.model else paths.checkpoint_dir / 'latest_model.pt'

    print(f'Target dir: {args.target}')
    dur_target_dir, pred_target_dir = Path(args.target) / 'durations', Path(args.target) / 'predictions'
    dur_target_dir.mkdir(parents=True, exist_ok=True)
    pred_target_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model from {model_path}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = Aligner.from_checkpoint(checkpoint).eval().to(device)
    print(f'Loaded model with step {model.get_step()} on device: {device}')

    config, audio, symbols = checkpoint['config'], Audio(**config['audio']), checkpoint['symbols']
    data_symbols = unpickle_binary(paths.data_dir / 'symbols.pkl')
    assert data_symbols == symbols, 'Symbols from dataset do not match symbols from loaded model!'
    tokenizer = Tokenizer(symbols)
    dataloader = new_dataloader(dataset_path=paths.data_dir / 'dataset.pkl', mel_dir=paths.mel_dir,
                                token_dir=paths.token_dir, batch_size=args.batch_size)

    print(f'Performing STT model inference...')
    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        tokens, mel, tokens_len, mel_len = to_device(batch, device)
        pred_batch = model(mel)
        for b in range(tokens.size(0)):
            this_mel_len = mel_len[b]
            pred = pred_batch[b, :this_mel_len, :]
            pred = torch.softmax(pred, dim=-1)
            pred = pred.detach().cpu().numpy()
            item_id = batch['item_id'][b]
            np.save(pred_target_dir / f'{item_id}.npy', pred, allow_pickle=False)

    print(f'Extracting durations...')
    dataset = unpickle_binary(paths.data_dir / 'dataset.pkl')
    token_pred_files = []
    for item in dataset:
        file_name =  item['item_id'] + '.npy'
        token_file, pred_file = paths.token_dir / file_name, pred_target_dir / file_name
        token_pred_files.append((item, token_file, pred_file))

    pool = Pool(processes=args.num_workers)
    mapper = pool.imap_unordered(extract_durations_for_item, token_pred_files)
    for i, (item, durations) in tqdm.tqdm(enumerate(mapper), total=len(token_pred_files)):
        item_id = item['item_id']
        np.save(dur_target_dir / f'{item_id}.npy', durations, allow_pickle=False)