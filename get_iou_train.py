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
    sequence_dict = {}
    total_iou = {}
    for file_id in tqdm(os.listdir("../train/labels")):
        file_id = file_id.replace(".json","")
        text = text_dict[file_id]
        target = np.array(tokenizer(text))
        
        pred = np.load(f"output/predictions/{file_id}.npy")
        
        target_len = target.shape[0]
        pred_len = pred.shape[0]
        
        durations_beam, sequences = extract_durations_beam(target, pred, 5)
        sequence_dict[file_id] = tokenizer.decode(target[sequences[0][0]])
        tmp = np.cumsum(np.pad(durations_beam[0], (1, 0)))
        with open(f"../train/labels-convert/{file_id}.json",'r') as f:
            result = json.load(f)
        sent_iou = []
        for seg in result:
            for word in seg['l']:
                try:
                    x = 100
                    if tmp[word['st']]-tmp[word['st']-1]<x:
                        word['ps'] = int(tmp[word['st']-1]*20)
                    else:
                        word['ps'] = int((tmp[word['st']]-x)*20)
                    word['pe'] = int((tmp[word['et']]+tmp[word['et']+1])*10)
                except:
                    word['ps'] = 0
                    word['pe'] = 0
                    print(file_id)
                iou = get_iou([word['s'],word['e']],[word['ps'],word['pe']])
                sent_iou.append(iou)
        with open(f"../train/labels-convert/{file_id}.json",'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        total_iou[file_id] = sum(sent_iou)/len(sent_iou)
    print(sum(total_iou.values())/len(total_iou))
    with open(f"train_iou.json",'w') as f:
        json.dump(total_iou, f, indent=4, ensure_ascii=False)
    with open(f"sequence_dict.json",'w') as f:
        json.dump(sequence_dict, f, indent=4, ensure_ascii=False)
        