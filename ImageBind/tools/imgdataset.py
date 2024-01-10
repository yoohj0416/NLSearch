from pathlib import Path

from torch.utils.data import Dataset, DataLoader


class oneImageDataset(Dataset):

    def __init__(self, img_dir):
        
        self.img_dir = Path(img_dir)

        self.img_ids = []
        self.img_paths = []
        for img_path in self.img_dir.iterdir():
            self.img_ids.append(img_path.stem)
            self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        return {'img_id': self.img_ids[index], 'img_path': str(self.img_paths[index])}
    

if __name__ == '__main__':
    img_dir = '/home/hojin/data_archive/bdd100k/images/100k/val'
    train_dataset = oneImageDataset(img_dir)
    loader = DataLoader(train_dataset,
                        batch_size=10)
    
    for i, batch in enumerate(loader):
        print(batch)
        exit(0)