import json
import torch
from torch.utils.data import DataLoader, Dataset

class XRayDataset(Dataset):
    def __init__(self, annotations_file):
        with open(annotations_file, 'r') as f:
            self.img_labels = json.load(f)

    def __getitem__(self, index):
        img_path_1 = self.img_labels[index]['image_path_1']
        img_path_2 = self.img_labels[index]['image_path_2']
        report = self.img_labels[index]['report']
        anatomical_parts = self.img_labels[index]['anatomical_parts']
        medical_terms = self.img_labels[index]['medical_terms']
        return {
            'img_path_1': img_path_1,
            'img_path_2': img_path_2,
            'report': report,
            'anatomical_parts': anatomical_parts,
            'medical_terms': medical_terms
        }

    def __len__(self):
        return len(self.img_labels)

def collate_fn(batch):
    # Initialize empty dictionary to collect the batched data
    batched_data = {}
    # Iterate over keys
    for key in batch[0].keys():
        if isinstance(batch[0][key], list):
            # Handle variable-length lists (e.g., medical_terms, anatomical_parts)
            batched_data[key] = [item[key] for item in batch]
        else:
            # Handle fixed size elements (e.g., img_path, report)
            batched_data[key] = [item[key] for item in batch]
    return batched_data

def get_dataloader(annotations_file=r"E:\HKRG\Data\iu_xray\pre_train.json", batch_size=4):
    dataset = XRayDataset(annotations_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader
