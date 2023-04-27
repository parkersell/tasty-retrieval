We have included the index code from BEiT3 adapted for tasty videos and flicker dataset. We created the following files from scratch
1. create_karpathy_split.py -> used to turn tasty-videos into a image-retrieval task for finetuning
2. utils.py -> used for loading the data in tasty-videos
3. inference.py -> saves the csv files containing the predicted results k1 and the actual results id
4. analyze_results.iypnb -> contains my results for experimenting with sampling strategy and context prompting

We also modified several files from the BEiT3 codebase, but with only slight modifications so they are not included. 