from core_datautils import code2impath, json, tqdm
import torch
from PIL import Image
from PIL import ImageDraw
import random
import os
PER_IMAGE_OBJECTS_PATH = './objects2img_complete.json'
PER_IMAGE_TEST_PATH = '/home/amolina/DEWGraph/src/objects2image_test.json'
class ObjectSpecificDateLoader(torch.utils.data.Dataset):
    def __init__(self, df, object: str, transforms, min_date = 1930, max_date = 1999, freq = 10, evaluate = False):
        super().__init__()

        with open(PER_IMAGE_OBJECTS_PATH if not evaluate else PER_IMAGE_TEST_PATH, 'r') as f:
            elements = json.load(f)[object]

        self.freq = freq
        self.df = df
        self.available_labels = list(range(min_date, max_date + 1, self.freq))
        self.category = object

        base_folder = f"/data/113-2/users/amolina/cir_date/objects/{object}/"
        ckpt_path = base_folder + ("data_ckpt.json" if not evaluate else "evaluation_data_ckpt.json")
        errors = 0

        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as f:
                self.datum = json.load(f)
        else:
            self.datum = []
            for image in tqdm(elements, desc = f'creating dataset for object: {object}...'):
                impath = code2impath(image)
                jsonpath = impath.replace('.jpg', '_v3.json')

                with open(jsonpath, 'r') as f:
                    data = json.load(f)

                for classname, bbx  in zip(data['detection_class_entities'], data['detection_boxes']):
                    if classname == object and (bbx[2] - bbx[0]) * (bbx[3] - bbx[1]) > (10000 / 4) /(224*224): # From the original paper
                        # https://github.com/cesc47/DEXPERT/blob/main/src/datasets.py#L145 In my experience 10.000 is a bit exagerated
                        try:
                            self.datum.append({
                                'impath': impath,
                                'bbox': bbx,
                                'year': int(df[df['code'] == int(image)]['year'].iloc[0])
                            })
                        except IndexError as e:
                            errors += 1
            

            os.makedirs(base_folder, exist_ok = True)
            with open(ckpt_path, 'w') as f:
                json.dump(self.datum, f)
        print(f"Dataset for {object} loaded with {len(self.datum)} samples and {errors} errors")
        self.es_trans = transforms


    def __len__(self):
        return len(self.datum)

    def sample(self, idx):
        data = self.datum[idx]

        image = Image.open(data['impath']).convert("RGB")
        W, H = image.size

        x1, y1, x2, y2 = data["bbox"]

        left = int(x1 * W)
        top = int(y1 * H)
        right = int(x2 * W)
        bottom = int(y2 * H)
        draw = ImageDraw.Draw(image)
        draw.rectangle((left, top, right, bottom), outline="red", width=3)
        image.save(f"{idx}.png")

        print(data)

    def __getitem__(self, idx):
        data = self.datum[idx]

        image = Image.open(data['impath']).convert("RGB")
        W, H = image.size

        x1, y1, x2, y2 = data["bbox"]

        left = int(x1 * W)
        top = int(y1 * H)
        right = int(x2 * W)
        bottom = int(y2 * H)

        crop = image.crop((left, top, right, bottom)).resize((224, 224))
        year = data['year']

        return self.es_trans(crop), self.available_labels.index(year - (year % self.freq)), self.category


class SpecialistDataloaderWithClass(ObjectSpecificDateLoader):
    def add(self, dataset):
        return SummedDataset(self, dataset)
    def __add__(self, dataset):
        return self.add(dataset)
    def get_one_sample(self, idx):
        data = self.datum[idx]

        image = Image.open(data['impath']).convert("RGB")
        W, H = image.size

        x1, y1, x2, y2 = data["bbox"]

        left = int(x1 * W)
        top = int(y1 * H)
        right = int(x2 * W)
        bottom = int(y2 * H)

        crop = image.crop((left, top, right, bottom)).resize((224, 224))
        year = data['year']

        # Aquí fem per dècada perque si no tindrem molts pocs samples
        return self.es_trans(crop), self.available_labels.index(year - (year % 10)), self.category

    def __getitem__(self, item):
        image_A, condition_B, category = self.get_one_sample(item)
        image_B, condition_A, _ = self.get_one_sample(random.randint(0, len(self.datum) - 1))

        # We need to enforce that:
        # image_A + condition_A = image_B
        # image_B + condition_B = image_A
        return image_A, condition_A, image_B, condition_B, category


class SummedDataset:
    def __init__(self, dataset_left, dataset_right) -> None:
        self.left = dataset_left
        self.right = dataset_right

    def __len__(self):
        return len(self.left) + len(self.right)

    def __getitem__(self, idx):
        if idx > (len(self.left) - 1):
            idx_corrected = idx % len(self.left)
            return self.right[idx_corrected]

        return self.left[idx]

    def add(self, dataset):
        return SummedDataset(self, dataset)
    def __add__(self, dataset):
        return self.add(dataset)


