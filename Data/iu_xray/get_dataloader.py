import json
from torch.utils.data import DataLoader, Dataset

class XRayDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        img_path_1 = self.data[index].get('image_path_1', '')
        img_path_2 = self.data[index].get('image_path_2', '')
        report = self.data[index].get('report', '')
        anatomical_parts = self.data[index].get('anatomical_parts', '')
        medical_terms = self.data[index].get('medical_terms', '')
        return {
            'img_path_1': img_path_1,
            'img_path_2': img_path_2,
            'report': report,
            'anatomical_parts':anatomical_parts,
            'medical_terms': medical_terms
        }

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    batched_data = {}
    for key in batch[0].keys():
        batched_data[key] = [item[key] for item in batch]
    return batched_data


def get_dataloader(split, batch_size=4):
    with open(r'E:\HKRG\Data\iu_xray\train.json', 'r') as file:
        data = json.load(file)
    dataset = XRayDataset(data[split])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader





