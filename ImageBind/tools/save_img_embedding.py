import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pinecone
from pathlib import Path
import time
import argparse

from imagebind import data
import torch
from torch.utils.data import DataLoader
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from imgdataset import oneImageDataset


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', required=True, type=str, help='directory of images')
    parser.add_argument('--api', required=True, type=str, help='enter your pinecone API')
    parser.add_argument('-i', '--index_name', required=True, type=str, help='pinecone index name')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size with inference of one image')

    args = parser.parse_args()
    return args


def main():

    args = get_args()

    image_dataset = oneImageDataset(args.img_dir)
    image_loader = DataLoader(image_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False)

    # Make Pinecone index
    pinecone.init(api_key=args.api, environment='gcp-starter')
    pinecone.create_index(args.index_name, dimension=1024, metric='cosine')
    index = pinecone.Index(args.index_name)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    for i, batch in enumerate(image_loader):

        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(batch['img_path'], device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        image_features = embeddings[ModalityType.VISION]

        for j, image_feature in enumerate(image_features):
            index.upsert(vectors=[{'id': batch['img_id'][j],
                                   'values': image_feature.cpu().numpy()}])


if __name__ == '__main__':
    main()