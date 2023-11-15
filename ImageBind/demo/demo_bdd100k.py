import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pinecone
from pathlib import Path
import time

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def main():
    # image_paths = ["/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/8fd046f2-ecff0000.jpg",]
    image_paths = ["/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/8fd046f2-ecff0000.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/93843437-b74101a3.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/ff55861e-a06b953c.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/8dd5f9b7-00000000.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/84f35817-cffdd105.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/974e613c-decea44e.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/83d03917-00000000.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/8fc433c9-2baf153a.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/9c35f247-c1736063.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/9b52e4af-205f725b.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/905625db-00000000.jpg",
                   "/users/PAS2119/yoohj0416/NLSearch/data/bdd100k/images/10k/val/99ebed9d-b6dd1203.jpg"]
    
    plt.figure(figsize=(16,10))

    original_images = []
    names = []
    for image_path in image_paths:
        name = Path(image_path).name

        image = Image.open(image_path).convert("RGB")

        plt.subplot(4, 3, len(original_images) + 1)
        plt.imshow(image)
        plt.title(name)
        plt.xticks([])
        plt.yticks([])

        original_images.append(image)
        names.append(name)
    
    plt.tight_layout()
    plt.savefig('/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/bdd10k_sample.png')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Load data
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    # Calculating cosine similarity
    image_features = embeddings[ModalityType.VISION]

    pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")
    pinecone.create_index('imagesearch-demo', dimension=1024, metric='cosine')

    index = pinecone.Index('imagesearch-demo')

    for i, image_feature in enumerate(image_features):
        index.upsert(vectors=[{'id': Path(image_paths[i]).name, 'values': image_feature.numpy()}])

    # Wait time for completing update pinecone index
    time.sleep(10)

    # Image similarity search by text
    query_texts = ["time is night-time",
                   "a weather condition is snowy",
                   "a car is in a parking lot",
                   "a car is in a highway",
                   "this scene is about residential area",
                   "people are crossing a crosswalk",
                   "a car is currently stationary."]
    k = 3
    
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(query_texts, device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    plt.figure(figsize=(16, 18))
    text_features = embeddings[ModalityType.TEXT]
    for i, feature in enumerate(text_features):
        
        search_result = index.query(
            vector=[feature.tolist()],
            top_k=k,
        )

        for j, result in enumerate(search_result['matches']):
            
            image = original_images[names.index(result['id'])]
            plt.subplot(len(query_texts), k, i * k + j + 1)
            plt.imshow(image)
            plt.title(f'Top-{j + 1} image searched by \"{query_texts[i]}\"')
            plt.xticks([])
            plt.yticks([])
        
    plt.tight_layout()
    plt.savefig('/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/bdd10k_text_query.png')

    # Delete pinecone index
    pinecone.delete_index('imagesearch-demo')


if __name__ == '__main__':
    main()
