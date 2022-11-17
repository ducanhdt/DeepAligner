import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from dfa.audio import Audio
from dfa.duration_extraction import  extract_durations_beam

from dfa.text import Tokenizer
from dfa.utils import unpickle_binary
from dfa.utils import read_config
from dfa.paths import Paths
import numpy as np
def get_iou(ground_truth, pred):
    ix1 = np.maximum(ground_truth[0], pred[0])
    ix2 = np.minimum(ground_truth[1], pred[1])
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
    area_of_intersection = i_width
    gt_width = ground_truth[1] - ground_truth[0] + 1
    pd_width = pred[1] - pred[0] + 1
    area_of_union = gt_width + pd_width - area_of_intersection
    iou = area_of_intersection / area_of_union
    return iou
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    args = parser.parse_args()
    config = read_config(args.config)
    paths = Paths.from_config(config['paths'])

    symbols = unpickle_binary(paths.data_dir / 'symbols.pkl')
    # audio = Audio.from_config(config['audio'])
    tokenizer = Tokenizer(symbols)

    with open('text_dict.json','r') as f:
        text_dict= json.load(f)
    
    total_iou = {}
    for file_id in tqdm(os.listdir("../public_test_lyrics_json/public_test/json_lyrics-convert")):
        file_id = file_id.replace(".json","")
        text = text_dict[file_id]
        target = np.array(tokenizer(text))
        
        pred = np.load(f"output/predictions/{file_id}.npy")
        
        target_len = target.shape[0]
        pred_len = pred.shape[0]
        
        durations_beam, sequences = extract_durations_beam(target, pred, 5)
        tmp = np.cumsum(np.pad(durations_beam[0], (1, 0)))
        with open(f"../public_test_lyrics_json/public_test/json_lyrics-convert/{file_id}.json",'r') as f:
            result = json.load(f)
        sent_iou = []
        for seg in result:
            for i,word in enumerate(seg['l']):
                try:
                    seg['l'][i] = {
                            "d":word['d'],
                            "s":int(tmp[word['st']-1]*20),
                            "e":int(tmp[word['et']]*20),
                            }
                except:
                    word = {
                            "d":word['d'],
                            "s":0,
                            "e":0,
                            }

        with open(f"../public_test_lyrics_json/public_test/submit/{file_id}.json",'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        