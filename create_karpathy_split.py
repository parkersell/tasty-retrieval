import os
from utils import *
from typing import Dict, List
import numpy as np
from tqdm import tqdm

"""
One major difference between the two image retrieval datasets (COCO and Flickr30k) and TastyVideos is that for each image there are 5 sentences whereas
in TastyVideos, for each sentence (recipe_step) that are many images. 

However, this doesn't really effect inference in the slightest. Because the model is trained, there is nothing that would need to change, if you just gave less sentences per image. 

    You could break the sentence into it's semantic different phrases 
        ex:  'In a very large bowl, whisk together 12 cups (1.5 kg) of flour, ½ cup (100 g) of sugar, and 2 teaspoons of salt.' ->
            ['In a very large bowl, whisk together 12 cups (1.5 kg) of flour', '½ cup (100 g) of sugar', '2 teaspoons of salt.']

    But I am not sure this is entirely simple, this would require a different model for breaking up a sentence into phrases

If we were to do fine-tuning, we would need to maybe adjust the hyperparams to adjust for the difference in the data. Or we would just have to live with poorer result

Additional Thought: Do we want to save a video frame interval? (Just check there are not gaps in current frame_indices) 
    This would give us videos for a given query. However, this model doesn't support this yet. So can easily add in later 
    Conclusion: Don't do this


Process for Converting to Karpathy Split:
    1. Load all the images into one folder by replacing folder/image_num with folder-image_num (Do we need to?) -> Don't need to
    2. Create counters for sentence_ids and image_ids -> Create their own
    3. Break raw text into tokens (Does it even use this tokenizer?) -> Don't need to
    4. Load into format {"sentids":[NOT USED], "imgid": NOT USED, "sentences": {"tokens":[NOT USED], "raw": recipe_step, "imgid": NOT USED, "sentid":NOT USED}, "split":split, "filename":path}
    5. Iterate for selected images

Process for Selecting Images
    1. Take a constant number of images even spaced out for each recipe_step. Let's say 5
"""


def select_images(frames:np.ndarray, num_frames=1)->List:
    frames = frames.squeeze().tolist()

    if len(frames) > num_frames:
        # Add two so that they are farther from the end and the beginning
        indices = np.linspace(0, len(frames) - 1, num_frames+2, dtype=int)[1:-1]
        # Create a list of the selected frames
        frames = [frames[i] for i in indices]
    return frames

def prompt1_step(step, recipe):
    recipe_name = recipe['title'].replace('-', ' ')
    return f"In the recipe {recipe_name}, the person should {step}"

def prompt2_step(step, recipe):
    recipe_name = recipe['title'].replace('-', ' ')
    return f"In the recipe {recipe_name}, {step}"

def prompt3_step(step, recipe):
    recipe_name = recipe['title'].replace('-', ' ')
    return f"In the recipe {recipe_name} with the ingredients {','.join(recipe['ingredients_names'])}, {step}"


def add_recipe(recipe:Dict, dataset:Dict):
    for i, step in enumerate(recipe['recipe_steps']):
        images = select_images(np.array(recipe['frame_indices'][i]))

        for im in images:
            base_dict = {"sentences": [{"raw": ""}], "split": "", "filename": ""}
            base_dict['sentences'][0]['raw'] = prompt3_step(step, recipe)
            base_dict['split'] = 'test' # only reads in test in evaluation time
            base_dict['filename'] = os.path.join(TASTY_FRAMES_PATH, recipe['title'], 'frames', "{:05d}.jpg".format(im))
            dataset['images'].append(base_dict)

def main():
    recipes = load_recipes(RECIPE_LIST_PATH)
    dataset = {'images':[], 'dataset':'tasty'}
    skipped = []
    for i, recipe_path in enumerate(tqdm(recipes)):
        if i > 1000: break
        try:
            recipe = load_recipe(recipe_path)
        except FileNotFoundError:
            skipped.append(recipe_path)
        add_recipe(recipe, dataset)
    
    print(skipped)
    save_dataset(dataset, 'dataset_1000_tasty_1_image_prompt3')
        



if __name__ == "__main__":
    main()
    # recipe = load_recipe('bleeding-vampire-drink')
    # print(recipe)