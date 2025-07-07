import torch
from tqdm import tqdm
import csv
import os
from Data.MIMIC_CXR.get_M_dataloader import get_dataloader
from Model.Generation.Model_G_HKRG import HKRGModel_G

def test_G(model, data_loader, criterion, device, csv_path):
    model.eval()
    total_loss = 0
    hypotheses = []

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Folder Path', 'Predicted Report'])

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluation", leave=False):
                images_batch_1 = batch['img_path_1'].to(device)
                images_batch_2 = batch['img_path_2'].to(device)
                reports = batch['report']
                anatomical_parts = batch['anatomical_parts'].to(device)
                medical_terms = batch['medical_terms'].to(device)

                outputs, targets = model(images_batch_1, images_batch_2, reports, anatomical_parts, medical_terms)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                predicted_ids = torch.argmax(outputs, dim=-1).view(len(reports), -1)
                decoded_predictions = model.text_processor.decode_text(predicted_ids)

                folder_paths = [os.path.dirname(img_path) for img_path in batch['img_path_1']]

                for folder_path, prediction in zip(folder_paths, decoded_predictions):
                    csvwriter.writerow([folder_path, prediction])

                hypotheses.extend(decoded_predictions)

    total_loss /= len(data_loader)
    print(f"Test Loss: {total_loss}")
    return total_loss, hypotheses

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ''  # Model address
    model = HKRGModel_G()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    test_loader = get_dataloader('test')
    criterion = torch.nn.CrossEntropyLoss()
    save_path = r'E:\HKRG\Data\MIMIC_CXR\res_labeled.csv'
    test_loss, _ = test_G(model, test_loader, criterion, device, save_path)
