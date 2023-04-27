
import torch
import numpy as np
from transformers import XLMRobertaTokenizer
import pandas as pd
from tqdm import tqdm
def infer(dataset, model, device, handler):
    scores = torch.load('scores.pt')
    iids = torch.load('iids.pt')
    tiids = torch.load('tiids.pt')
    print(iids, tiids)
    print(dataset[iids[0]])
    print(dataset)
    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]
    print(topk1_iids.size())
    tokenizer = XLMRobertaTokenizer("/home/parker-alien/Documents/CV-586/beit3.spm")
    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0]
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0]
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0]
    ir_np = ["id,success,k1,step,video_name,image_num,image_path"]
    for i, val in enumerate(tqdm(ir_r1.cpu().numpy())):
        image_path:str = dataset[tiids[i]]['image_path']
        video_name, image_num = image_path.replace("dataset[tiids[i]]['image_path']", "").replace(".jpg", "").split("/frames/")
        values = list(map(str, [tiids[i].item(),round(val),topk1_iids[0][i].item(),tokenizer.decode(dataset[tiids[i]]['language_tokens'], skip_special_tokens=True).replace (",", " "),video_name,image_num,image_path]))
        ir_np.append(",".join(values))
    with open('ir_r1_1_image_1000.csv', 'w') as f:
        f.write("\n".join(ir_np))
    