import torch
import logging
import os
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from Model.Generation.Model_G_HKRG import HKRGModel_G
from Data.MIMIC_CXR.get_M_dataloader import get_dataloader
from Metrics.NLG_Metrics.NLG_M import calculate_metrics
import torch.optim.lr_scheduler as lr_scheduler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
writer = SummaryWriter(r'E:\HKRG\Train\runs\MIMIC-CXR\training')
torch.autograd.set_detect_anomaly(True)


def test_G(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    references = []
    hypotheses = []

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

            # Decode targets and outputs for metric calculation
            predicted_ids = torch.argmax(outputs, dim=-1).view(len(reports), -1)
            targets = targets.view(len(reports), -1)

            decoded_targets = model.text_processor.decode_text(targets)
            decoded_predictions = model.text_processor.decode_text(predicted_ids)

            for i, (target, prediction) in enumerate(zip(decoded_targets, decoded_predictions)):
                print(f"Original Report {i + 1}: {target}")
                print(f"Predicted Report {i + 1}: {prediction}\n")

            hypotheses.extend(decoded_predictions)
            references.extend([[ref] for ref in decoded_targets])

    total_loss /= len(data_loader)
    metrics = calculate_metrics(references, hypotheses)

    return total_loss, metrics


def val_G(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    references = []
    hypotheses = []

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

            # Decode targets and outputs for metric calculation
            predicted_ids = torch.argmax(outputs, dim=-1).view(len(reports), -1)
            targets = targets.view(len(reports), -1)

            decoded_targets = model.text_processor.decode_text(targets)
            decoded_predictions = model.text_processor.decode_text(predicted_ids)

            for i, (target, prediction) in enumerate(zip(decoded_targets, decoded_predictions)):
                print(f"Original Report {i + 1}: {target}")
                print(f"Predicted Report {i + 1}: {prediction}\n")

            hypotheses.extend(decoded_predictions)
            references.extend([[ref] for ref in decoded_targets])

    total_loss /= len(data_loader)
    metrics = calculate_metrics(references, hypotheses)
    return total_loss, metrics



def train_G(model, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.8, 0.999), weight_decay=0.999)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    train_loader = get_dataloader('train')
    val_loader = get_dataloader('val')
    test_loader = get_dataloader('test')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    best_metric = float('-inf')
    best_test_metric = float('-inf')

    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for i, batch in progress_bar:
            images_batch_1 = batch['img_path_1']
            images_batch_2 = batch['img_path_2']
            reports = batch['report']
            anatomical_parts = batch['anatomical_parts']
            medical_terms = batch['medical_terms']

            optimizer.zero_grad()

            outputs, targets = model(images_batch_1, images_batch_2, reports, anatomical_parts, medical_terms)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()

            if (i + 1) % 5 == 0:
                val_loss, val_metrics = val_G(model, val_loader, criterion, device)
                logger.info(f"Validation Loss: {val_loss}")
                logger.info(f"Validation Metrics: {val_metrics}")

                writer.add_scalar('Loss/val', val_loss, epoch * len(train_loader) + i)
                writer.add_scalars('Metrics/val', val_metrics, epoch * len(train_loader) + i)

                current_metric = (val_metrics['BLEU-4'] + val_metrics['ROUGE-L'] + val_metrics['METEOR']) / 3

                if current_metric > best_metric:
                    best_metric = current_metric
                    torch.save(model.state_dict(), os.path.join('model','best_model_val.pth'))

                scheduler.step(val_loss)

            if (i + 1) % 10 == 0:
                test_loss, test_metrics = test_G(model, test_loader, criterion, device)
                logger.info(f"Test Loss: {test_loss}")
                logger.info(f"Test Metrics: {test_metrics}")

                writer.add_scalar('Loss/test', test_loss, epoch * len(train_loader) + i)
                writer.add_scalars('Metrics/test', test_metrics, epoch * len(train_loader) + i)

                test_metric = (test_metrics['BLEU-4'] + test_metrics['ROUGE-L'] + test_metrics['METEOR']) / 3

                if test_metric > best_test_metric:
                    best_test_metric = test_metric
                    torch.save(model.state_dict(), os.path.join('model','best_model_test.pth'))

            progress_bar.set_postfix(loss=epoch_loss)

        logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader)}")
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)

    writer.close()
    logger.info("Training complete.")


if __name__ == '__main__':
    model = HKRGModel_G()
    train_G(model, epochs=500)
