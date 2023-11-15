import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def main():
    text_list = ['train', 'horse', 'dog', 'airplane', 'bike', 'stop_sign', 'bed', 'cat', 'bus', 'car']
    # text_list = ['bike', 'car', 'stop_sign']
    image_paths = ['/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/5898935548_0221a115a6_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/6286981167_07746c99a3_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/7213399710_5435238cfc_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/7772698096_c771bbeb34_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/27264220_94192556a1_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/3290918536_945aaf04ba_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/4147178335_af0b4831bf_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/3440089012_0e580cecc7_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/3330877365_42c8d3e29d_z.jpg',
                   '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/131678323_594a1bd9f9_z.jpg',]
    # image_paths = ['/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/27264220_94192556a1_z.jpg',
    #                '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/131678323_594a1bd9f9_z.jpg',
    #                '/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/sample_images/3290918536_945aaf04ba_z.jpg']
    
    original_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        original_images.append(image)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    # Calculating cosine similarity
    image_features = embeddings[ModalityType.VISION]
    text_features = embeddings[ModalityType.TEXT]
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features @ image_features.T

    count = len(text_list)

    plt.figure(figsize=(16, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(count), text_list, fontsize=18)
    plt.xticks([])

    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
    plt.tight_layout()

    plt.savefig('/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/similarity.png')

    # Zero-shot image classification
    text_probs = torch.softmax(100.0 * image_features @ text_features.T, dim=-1)
    top_probs, top_labels = text_probs.topk(5, dim=-1)
    
    plt.figure(figsize=(16, 16))
    for i, image in enumerate(original_images):
        plt.subplot(5, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(5, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [text_list[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()

    plt.savefig('/users/PAS2119/yoohj0416/NLSearch/ImageBind/demo/zero-shot.png')

    # print(
    #     "Vision x Text: ",
    #     torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    # )

if __name__ == '__main__':
    main()