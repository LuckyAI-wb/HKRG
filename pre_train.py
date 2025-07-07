import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import os
from Utils.SmoothedValue import SmoothedValue
from Model.HKRG_Pre.Model_Pre_HKRG import HKRGModel
from Data.iu_xray.get_dataloader_proprcess import get_dataloader
from loss.cl_loss import ClipLoss
from loss.ITM import ITM_task, ITMLoss


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
writer = SummaryWriter(r'E:\HKRG\Pre_logger\runs\pre_training')

torch.autograd.set_detect_anomaly(True)

def train(model, data_loader, epochs):
    model.train()
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-5},
    ])

    smoothed_loss_clip = SmoothedValue(window_size=50)
    smoothed_loss_mae = SmoothedValue(window_size=50)
    smoothed_loss_mlm = SmoothedValue(window_size=50)
    smoothed_loss_itm = SmoothedValue(window_size=50)
    smoothed_loss_itm_me = SmoothedValue(window_size=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Itm_loss_func = ITMLoss
    clip_loss_func = ClipLoss()
    mae_loss_func = torch.nn.MSELoss()
    mlm_loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}",
                            leave=False)

        for i, batch in progress_bar:
            images_batch_1 = batch['img_path_1']
            images_batch_2 = batch['img_path_2']
            reports = batch['report']
            anatomical_parts = batch['anatomical_parts']
            medical_terms = batch['medical_terms']

            optimizer.zero_grad()

            labels_batch_an = ITM_task(images_batch_1, anatomical_parts)
            labels_batch_an = labels_batch_an.to(device)

            labels_batch_me = ITM_task(images_batch_1, medical_terms)
            labels_batch_me = labels_batch_me.to(device)

            # Pass the whole batch through the model
            vision_features_clip, text_features_clip, vision_features_itm_an, text_features_itm_an, vision_features_itm_me, text_features_itm_me, reconstructed_image, images_1, text_features_MLM, text_labels = model(images_batch_1, images_batch_2, reports,
                                                                                                                                                        anatomical_parts,medical_terms)
            # Compute losses directly on batch outputs
            loss_mae = mae_loss_func(reconstructed_image, images_1)
            loss_mlm = mlm_loss_func(text_features_MLM, text_labels)
            loss_clip = clip_loss_func(vision_features_clip, text_features_clip)
            loss_ITM_an = Itm_loss_func(vision_features_itm_an, text_features_itm_an, labels_batch_an)
            loss_ITM_me = Itm_loss_func(vision_features_itm_me, text_features_itm_me, labels_batch_me)

            # Combine losses
            total_loss = loss_clip + loss_mae + loss_mlm + loss_ITM_an + loss_ITM_me
            total_loss.backward()
            optimizer.step()

            # Update progress and log losses
            total_loss_value = total_loss.item()
            smoothed_loss_clip.update(loss_clip.item())
            smoothed_loss_mae.update(loss_mae.item())
            smoothed_loss_mlm.update(loss_mlm.item())
            smoothed_loss_itm.update(loss_ITM_an.item())
            smoothed_loss_itm_me.update(loss_ITM_me.item())

            progress_bar.set_postfix(loss=total_loss_value, clip_loss=smoothed_loss_clip.global_avg,
                                     mae_loss=smoothed_loss_mae.global_avg, mlm_loss=smoothed_loss_mlm.global_avg, itm_an_loss=smoothed_loss_itm.global_avg,
                                     itm_me_loss=smoothed_loss_itm_me.global_avg)

            # TensorBoard logging
            writer.add_scalar('Loss/Total', total_loss_value, epoch * len(data_loader) + i)
            writer.add_scalar('Loss/Clip', smoothed_loss_clip.global_avg, epoch * len(data_loader) + i)
            writer.add_scalar('Loss/MAE', smoothed_loss_mae.global_avg, epoch * len(data_loader) + i)
            writer.add_scalar('Loss/MLM', smoothed_loss_mlm.global_avg, epoch * len(data_loader) + i)
            writer.add_scalar('Loss/ITM_an', smoothed_loss_itm.global_avg, epoch * len(data_loader) + i)
            writer.add_scalar('Loss/ITM_me', smoothed_loss_itm_me.global_avg, epoch * len(data_loader) + i)
            epoch_loss += total_loss_value

        # End of epoch handling
        avg_epoch_loss = epoch_loss / len(data_loader)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), os.path.join('model', 'best_model.pth'))
            logger.info(f"New best model saved with loss: {best_loss:.4f}")
        torch.save(model.state_dict(), os.path.join('model', f'model_epoch_{epoch + 1}.pth'))
        logger.info(f"Model saved for epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4f}")

    writer.close()
    logger.info("Training complete.")


if __name__ == '__main__':
    model = HKRGModel()
    data_loader = get_dataloader()
    train(model, data_loader, epochs=1000)
