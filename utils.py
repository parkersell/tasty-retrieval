import pickle
import os
import json


TASTY_FRAMES_PATH = '/mnt/e/TastyFrames'
TASTY_DIR = '/mnt/e/Tasty_dir'
TASTY_SENTENCES_PATH = os.path.join(TASTY_DIR, 'DATASET_tasty')
RECIPE_LIST_PATH = os.path.join(TASTY_DIR, 'DATA', 'TASTY_splits', 'ALL_RECIPES.txt')
OUTPUT_FOLDER = '/home/parker-alien/Documents/CV-586/tasty_data'

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_recipes(path):
    with open(path, 'r') as f:
        recipes = f.read().splitlines()
    return recipes

def load_recipe(name):
    return load_pickle(os.path.join(TASTY_SENTENCES_PATH, f'{name}.pkl'))

def save_dataset(dataset, file_name='dataset_tasty'):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    with open(os.path.join(OUTPUT_FOLDER,'%s.json' % file_name), 'w') as f:
        json.dump(dataset, f)

