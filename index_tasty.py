import sys
sys.path.insert(0, '/home/parker-alien/Documents/CV-586/unilm/beit3')
from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/home/parker-alien/Documents/CV-586/beit3.spm")
data_path = '/home/parker-alien/Documents/CV-586/tasty_data'
RetrievalDataset.make_tasty_dataset_index(
    data_path=data_path,
    tokenizer=tokenizer,
    file_name='dataset_1000_tasty_1_image_prompt3',
)