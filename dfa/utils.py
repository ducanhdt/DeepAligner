import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Union

import torch
import yaml

sss = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~’“”…'
def remove_punc(a_string: str)-> str:
    n_string = [a_string[0]]
    for c in a_string:
        if c != n_string[-1]:
            n_string.append(c)
    a_string = ''.join(n_string)
    return a_string.translate(str.maketrans('', '', sss))

def read_metafile(path: str) -> Dict[str, str]:
    text_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            text_id, text = split[0], split[-1]
            text_dict[text_id] = text.strip()
    return text_dict

def read_lyric(train_path:str,predict_path = None) ->Dict[str, str]:
    text_dict = {}
    for path in [train_path,predict_path]:
        file_list = os.listdir(path)
        Path(path+'-convert').mkdir(parents=True, exist_ok=True)
        for file in file_list:
            with open(f"{path}/{file}",'r') as f:
                data = json.load(f)
                # text = ' '+remove_punc(' '.join([i["d"] for s in data for i in s['l']])).lower()+' '
                text = ''
                for seg in data:
                    for w in seg['l']:
                        norm_d = ' '.join(remove_punc(w['d'].lower()).split())
                        if norm_d:
                            w['st'] = len(text)+1
                            text += ' '+norm_d
                            w['et'] = len(text)
                            w['norm_d'] = norm_d
                        else:
                            print(file)
                            w['st'] = len(text)-1
                            w['et'] = len(text)
                            w['norm_d'] = norm_d
                with open(f'{path}-convert/{file}','w') as f:
                    json.dump(data,f,ensure_ascii=False,indent=4)
                text_dict[file.replace('.json','')] = text + ' '
    
    return text_dict

def read_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, 'w+', encoding='utf-8') as stream:
        yaml.dump(config, stream, default_flow_style=False)


def get_files(path: str, extension='.wav') -> List[Path]:
    return list(Path(path).expanduser().resolve().rglob(f'*{extension}'))


def pickle_binary(data: object, file: Union[str, Path]) -> None:
    with open(str(file), 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]) -> Any:
    with open(str(file), 'rb') as f:
        return pickle.load(f)


def to_device(batch: dict, device: torch.device) -> tuple:
    tokens, mel, tokens_len, mel_len = batch['tokens'], batch['mel'], \
                                       batch['tokens_len'], batch['mel_len']
    tokens, mel, tokens_len, mel_len = tokens.to(device), mel.to(device), \
                                       tokens_len.to(device), mel_len.to(device)
    return tokens, mel, tokens_len, mel_len