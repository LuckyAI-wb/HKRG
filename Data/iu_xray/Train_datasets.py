import os
import json
import pandas as pd

# Load the dataset
with open('annotation.json', 'r') as file:
    data = json.load(file)

# Load the anatomical parts and medical terms
anatomical_parts_df = pd.read_csv('anatomical_parts.csv')
medical_terms_df = pd.read_csv('medical_terms.csv')

# Convert the anatomical parts and medical terms to sets for faster lookup
anatomical_parts_set = set(anatomical_parts_df['Anatomical Parts'].str.lower())
medical_terms_set = set(medical_terms_df['Medical Terms'].str.lower())

# Base directory to add to image paths
base_directory = 'E:\\HKRG\Data\\iu_xray\\images\\'

# Initialize a list to hold processed data
processed_data = {
    'train': [],
    'val': [],
    'test': []
}

# Process each entry and add new labels
for split in data:
    for entry in data[split]:
        report = entry.get('report', '')
        image_paths = entry.get('image_path', [])

        anatomical_parts = list(anatomical_parts_set)
        medical_terms = list(medical_terms_set)

        # Create new entry with image_path_1 and image_path_2 if applicable
        new_entry = {
            'report': report,
            'anatomical_parts': anatomical_parts,
            'medical_terms': medical_terms
        }

        if len(image_paths) > 0:
            new_entry['image_path_1'] = os.path.join(base_directory, image_paths[0]).replace('/', '\\')
        if len(image_paths) > 1:
            new_entry['image_path_2'] = os.path.join(base_directory, image_paths[1]).replace('/', '\\')

        processed_data[split].append(new_entry)

# Save the new dataset
with open('train.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print("New dataset with anatomical parts and medical terms labels has been created as 'train.json'.")
