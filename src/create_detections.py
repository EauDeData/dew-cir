import os
import json
import torch
import pandas as pd
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm import tqdm

# Paths
base_dew = '/data/113-2/users/amolina/DEW/DEW/DEW/Date_Estimation_in_the_Wild/'
dew_train_csv = os.path.join(base_dew, 'gt_test_ok.csv')

df = pd.read_csv(dew_train_csv, names=['year', 'code'])
iseven = input('Press "y" to process evens. "N" otherwise: ')
if iseven == 'y':
    df = df[df['code'].astype(str).str[0].astype(int) % 2 == 0]
elif iseven == 'n':
    df = df[df['code'].astype(str).str[0].astype(int) % 2 != 0]
else:
    raise  ValueError('Please enter "y" or "n"')

def code2impath(code, base_dew=base_dew):
    code = str(code)
    return os.path.join(base_dew, f"images/{code[0]}/{code[1:3]}/{code}.jpg")

# Load target labels from objects2img.json
with open('objects2img_keep.json', 'r') as f:
    objects2img = json.load(f)

target_labels = list(objects2img.keys())
text_queries = [target_labels]  # OWLv2 expects list of lists

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble", use_fast = True)
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").to(device)

model.eval()

CONFIDENCE_THRESHOLD = 0.2
BATCH_SIZE = 4

image_paths = [code2impath(code) for code in df['code']]
# Filter out already processed images
image_paths = [p for p in image_paths if not os.path.exists(p.replace('.jpg', '_v3.json'))]

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    opaths = [img_path.replace('.jpg', '_v3.json') for img_path in batch_paths]
    images = [Image.open(p).convert('RGB') for p in batch_paths ]

    # OWLv2 expects one text query list per image in the batch
    batch_text_queries = text_queries * len(images)
    inputs = processor(text=batch_text_queries, images=images, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([img.size[::-1] for img in images], device=device)
    batch_results = processor.post_process_object_detection(
        outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes
    )

    for img_path, image, results, output_path in zip(batch_paths, images, batch_results, opaths):

        boxes = results['boxes'].cpu().tolist()
        scores = results['scores'].cpu().tolist()
        labels = results['labels'].cpu().tolist()

        w, h = image.size
        rel_boxes = [
            [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
            for b in boxes
        ]

        result_dict = {
            "detection_class_entities": [target_labels[l] for l in labels],
            "detection_boxes": rel_boxes,
            "detection_scores": scores
        }

        # output_path = img_path.replace('.jpg', '_v3.json')

        with open(output_path, 'w') as f:
            json.dump(result_dict, f)
