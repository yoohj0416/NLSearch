from pathlib import Path
import json
from tqdm import tqdm


def main():
    # file_path = Path('/home/hojin/data_archive/bdd100k/info/100k/train/00a0f008-3c67908e.json')
    file_path = Path('/home/hojin/data_archive/bdd100k/info/100k/train/00a0f008-a315437f.json')
    # file_path = Path('/home/hojin/data_archive/bdd100k/info/100k/train/0ba6f2ab-060d43b9.json')

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        if len(data) == 0:
            print('file is None')
            return
        info = json.loads(data)

    if "gps" not in info:
        print("Field 'gps' not found.")
        return
    if len(info["gps"]) < 1:
        print("Empty trajectory data.")
        return

    print(info.keys())
    # print(len(info['gps']))
    # print(len(info['locations']))

    for i in range(len(info['gps'])):
        print(info['gps'][i])
    #     print(info['locations'][i])

    print(info['timelapse'])

    ####################################
    # Check number of timelapse videos #
    ####################################
    # info_dir = Path('/home/hojin/data_archive/bdd100k/info/100k/train')

    # file_cnt = 0
    # timelapse_cnt = 0
    # for file_path in tqdm(info_dir.iterdir(), total=len(list(info_dir.iterdir()))):

    #     with open(file_path, "r", encoding="utf-8") as f:
    #         data = f.read()
    #         if len(data) == 0:
    #             continue
    #         info = json.loads(data)

    #     if info['timelapse']:
    #         timelapse_cnt += 1
    #     file_cnt += 1
    
    # print(f"total num: {file_cnt:,}")
    # print(f"timelapse num: {timelapse_cnt:,}")
        

if __name__ == '__main__':
    main()