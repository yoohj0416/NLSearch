import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pinecone
from pathlib import Path
import time
import argparse
import json
from tqdm import tqdm

from imagebind import data
import torch
from torch.utils.data import DataLoader
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


classes_weather = ['clear', 'overcast', 'snowy', 'rainy', 
                   'partly cloudy', 'foggy']
classes_scene = ['city street', 'tunnel', 'highway', 'residential',
                 'parking lot', 'gas stations']


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', default='weather', choices=['weather', 'scene'], help='choose mode for tag classification')
    parser.add_argument('--api', required=True, type=str, help='enter your pinecone API')
    parser.add_argument('-i', '--index_name', required=True, type=str, help='pinecone index name')
    parser.add_argument('--label', required=True, type=str, help='path for bdd100k labels json file')

    args = parser.parse_args()
    return args


def main():

    args = get_args()

    pinecone.init(api_key=args.api, environment='gcp-starter')
    index = pinecone.Index(args.index_name)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    if args.mode == 'weather':
        query_texts = [f"a weather condition is {c}" for c in classes_weather]
    elif args.mode == 'scene':
        query_texts = [f"this image is about {c}" if c != "residential" else f"this image is about {c} area" for c in classes_scene]

    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(query_texts, device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    text_features = embeddings[ModalityType.TEXT].cpu().numpy()

    with open(args.label, 'r') as f:
        file_data = f.read()
    labels = json.loads(file_data)

    if args.mode == 'weather':
        total_num = {c: 0 for c in classes_weather}
        correct_num = {c: 0 for c in classes_weather}
    else:
        total_num = {c: 0 for c in classes_scene}
        correct_num = {c: 0 for c in classes_scene}

    for img_label in tqdm(labels):
        id = img_label['name'].split('.jpg')[0]
        if args.mode == 'weather':
            condition = img_label['attributes']['weather']
        else:
            condition = img_label['attributes']['scene']

        if condition == 'undefined':
            continue

        image_feature = index.query(id=id, include_values=True, top_k=1)['matches'][0]['values']
        image_feature = np.asarray(image_feature)

        similarity = image_feature @ text_features.T
        text_probs = torch.softmax(100.0 * torch.tensor(similarity), dim=-1)
        top_prob, top_label = text_probs.topk(1, dim=-1)

        if args.mode == 'weather':
            pred = classes_weather[top_label.numpy()[0]]
        else:
            pred = classes_scene[top_label.numpy()[0]]

        total_num[condition] += 1
        if pred == condition:
            correct_num[condition] += 1

    print(f"Accuracy of {args.mode} classification")
    for c in total_num.keys():
        print(f"{c}: {correct_num[c]} / {total_num[c]} ({correct_num[c] / total_num[c] * 100:.2f})")

        # print(text_probs)
        # print(text_probs.shape)
        # print(image_feature.shape)

        # exit(0)




if __name__ == '__main__':
    main()