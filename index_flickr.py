import sys
sys.path.insert(0, '/home/parker-alien/Documents/CV-586/unilm/beit3')
from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/home/parker-alien/Documents/CV-586/beit3.spm")
data_path = '/home/parker-alien/Documents/CV-586/data'
RetrievalDataset.make_flickr30k_dataset_index(
    data_path=data_path,
    tokenizer=tokenizer,
    karpathy_path=data_path,
)