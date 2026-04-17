import os
import pandas as pd
from tqdm import tqdm
import json
from multiprocessing import Pool

base_dew = '/data/113-2/users/amolina/DEW/DEW/DEW/Date_Estimation_in_the_Wild/'
dew_train_csv = os.path.join(base_dew, 'gt_train_ok.csv')
dew_test_csv = os.path.join(base_dew, 'gt_test_ok.csv')

df = pd.read_csv(os.path.join(dew_train_csv), names=['year', 'code'])
df_test = pd.read_csv(os.path.join(dew_test_csv), names=['year', 'code'])

def code2impath(code, base_dew = base_dew):
    code = str(code)
    return os.path.join(base_dew, f"images/{code[0]}/{code[1:3]}/{code}.jpg")

def list_object_items(dataframe):
    objects = set()
    images_with_objects = {}
    total_err = 0
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):

        json_path = code2impath(row['code']).replace('.jpg', '_v3.json')

        if os.path.exists(json_path):

            with open(json_path, 'r') as f:
                obj = json.load(f)['detection_class_entities']
                objects.update(obj)
                for obj in set(obj):
                    if not obj in images_with_objects:
                        images_with_objects[obj] = []
                    images_with_objects[obj].append(row['code'])

        else:
            total_err += 1
            continue
    print(f"Completed {len(sum(images_with_objects.values(), start = []))} out of {len(dataframe)} with {total_err} errors")
    return objects, images_with_objects



if __name__ == '__main__':
    objects, images_with_objects = list_object_items(df_test)
    imgs = {k: [int(x) for x in v] for k, v in images_with_objects.items()}
    json.dump(imgs, open('objects2image_test.json', 'w'), indent=3)

