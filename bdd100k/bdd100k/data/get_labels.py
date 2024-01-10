from pathlib import Path
import json


def main():
    # json_path = Path('/home/hojin/data_archive/bdd100k/labels/bdd100k_labels_images_train.json')
    json_path = Path('/home/hojin/data_archive/bdd100k/labels/bdd100k_labels_images_val.json')

    with open(json_path, "r", encoding="utf-8") as f:
        data = f.read()
        labels = json.loads(data)

    # for label in labels:
        # print(label['attributes']['timeofday'])

    # print(labels[0]['labels'])

    for label in labels[0]['labels']:
        print(label)

    # print(labels[0].keys())


if __name__ == '__main__':
    main()