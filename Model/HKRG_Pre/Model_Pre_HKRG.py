import torch
import string
import random
import re
from torch import nn
from Utils.safe_load_state_dict import safe_load_state_dict
from transformers import AutoTokenizer
from Model.Image_encoder import  SwinTransformer, SwinDecoder
from Model.Text_encoder import text_encoder
from torchvision import transforms
from PIL import Image
from Model.Cross_Modal_Fusion import MultiLayerEncoder
torch.autograd.set_detect_anomaly(True)

class HKRGModel(nn.Module):
    """
    Vision_model: swinv2_tiny_patch4_window8_256 (github)
    Text_mdoelL: Med-KEBERT (Hugging Face)
    """
    def __init__(self,
                 vision_model_path=r"",
                 text_model_path=r""):
        super(HKRGModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.vision_encoder = SwinTransformer().to(self.device)
            checkpoint = torch.load(vision_model_path, map_location=self.device)
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            safe_load_state_dict(self.vision_encoder, checkpoint)
            print("The visual encoder part has been successfully loaded, with partial weights if necessary.")
        except Exception as e:
            print("Error loading vision model.")
            print(str(e))

        try:
            # Text model and tokenizer initialization
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            self.text_encoder = text_encoder(text_model_path).to(self.device)
            print("Text encoder loaded successfully.")
        except Exception as e:
            print("Error loading text model.")
            print(str(e))

        self.MAEdecoder = SwinDecoder().to(self.device)
        self.cross_encoder=MultiLayerEncoder(768, 8, 6)
        self.final_classification = nn.Linear(768, 28895)
        self.conv= nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, padding=0)
        self.cross_conv = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, padding=1)
        self.self_attn = nn.MultiheadAttention(768, 8)
        self.adaptive_pool_text = nn.AdaptiveAvgPool2d((49, 768))

    def process_image(self, image_paths):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        images = [transform(Image.open(img_path).convert('RGB')) for img_path in image_paths]
        images = torch.stack(images)
        return images.to(self.device)

    def process_text(self, texts):
        processed_texts = []
        for text in texts:
            text = text.lower()
            text = re.sub(r'xxxx', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            processed_texts.append(text)
        return processed_texts

    def process_list_text(self, texts):
        sentences = [" ".join(item) for item in texts]
        return sentences

    def mask_image_patches(self ,image, mask_ratio=0.7, patch_size=16):
        assert image.size(2) % patch_size == 0 and image.size(
            3) % patch_size == 0, "Image dimensions must be divisible by the patch size."

        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(-1, image.size(1), patch_size, patch_size)

        total_patches = patches.size(0)
        num_to_mask = int(total_patches * mask_ratio)
        mask_indices = random.sample(range(total_patches), num_to_mask)

        full_mask = torch.ones_like(image)
        for idx in mask_indices:
            row = (idx // (image.size(3) // patch_size)) * patch_size
            col = (idx % (image.size(3) // patch_size)) * patch_size
            full_mask[:, :, row:row + patch_size, col:col + patch_size] = 0

        masked_image = image * full_mask

        return masked_image

    def mask_text_tokens(self, input_ids, mask_ratio=0.15):
        masked_input_ids = input_ids.clone()
        labels = input_ids.clone()

        batch_size, seq_length = input_ids.size()
        special_tokens_mask = []
        for i in range(batch_size):
            mask = self.tokenizer.get_special_tokens_mask(input_ids[i].tolist(), already_has_special_tokens=True)
            special_tokens_mask.append(mask)
        special_tokens_mask = torch.tensor(special_tokens_mask).bool().to(input_ids.device)

        tokens = [self.tokenizer.convert_ids_to_tokens(input_ids[i].tolist()) for i in range(batch_size)]

        def is_punctuation(token):
            return all(char in string.punctuation for char in token)

        for i in range(batch_size):
            valid_indices = [j for j, token in enumerate(tokens[i]) if
                             not special_tokens_mask[i, j] and not is_punctuation(token)]
            num_to_mask = max(1, int(len(valid_indices) * mask_ratio))
            masked_indices = random.sample(valid_indices, num_to_mask)

            # Apply masking
            for idx in masked_indices:
                masked_input_ids[i, idx] = self.tokenizer.mask_token_id

            # Prepare labels for masked language modeling
            labels[i, special_tokens_mask[i]] = -100  # Only compute loss on masked tokens
            labels[i] = torch.where(labels[i] >= self.tokenizer.vocab_size, -100, labels[i])
            labels[i] = torch.where(labels[i] < 0, -100, labels[i])

        return masked_input_ids, labels

    def attn(self, images,):

        image = self.process_image(images).to(self.device)
        if image is None:
            raise ValueError("Image processing failed.")
        _, vision_features_clip, attn = self.vision_encoder(image)

        return  attn

    def image_CLip_fusion(self,features_clip_1,features_clip_2):
        vision_features_clip = torch.stack([features_clip_1, features_clip_2], dim=2)
        vision_features_clip = self.conv(vision_features_clip)
        vision_features_clip = torch.squeeze(vision_features_clip, -1)
        return vision_features_clip

    def image_Cross_fusion(self, features_cross_1, features_cross_2):
        concatenated_tensor = torch.cat((features_cross_1, features_cross_2), dim=1).permute(0, 2, 1)
        vision_features_cross = self.cross_conv(concatenated_tensor)
        vision_features_cross = vision_features_cross.permute(0, 2, 1)
        return vision_features_cross

    def forward(self, images_1, images_2, texts, anatomical_parts, medical_terms):

        images_1 = self.process_image(images_1)
        if images_1 is None:
            raise ValueError("Image processing failed.")
        vision_features_cross_1, vision_features_clip_1, _ = self.vision_encoder(images_1)

        images_2 = self.process_image(images_2)
        if images_2 is None:
            raise ValueError("Image processing failed.")
        vision_features_cross_2, vision_features_clip_2, _ = self.vision_encoder(images_2)

        vision_features_clip = self.image_CLip_fusion(vision_features_clip_1, vision_features_clip_2)
        vision_features_cross = self.image_Cross_fusion(vision_features_cross_1, vision_features_cross_2)

        processed_texts = self.process_text(texts)
        inputs = self.tokenizer(processed_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        text_features_clip, text_features_cross, _ = self.text_encoder({'input_ids': input_ids, 'attention_mask': attention_mask})

        masked_input_ids, text_labels = self.mask_text_tokens(input_ids)
        _, text_features_MLM, _ = self.text_encoder({'input_ids': masked_input_ids, 'attention_mask': attention_mask})
        text_features_MLM = self.cross_encoder(text_features_MLM, vision_features_cross)
        text_features_MLM = text_features_MLM.view(-1, text_features_MLM.size(-1))
        text_features_MLM = self.final_classification(text_features_MLM)
        text_labels = text_labels.view(-1)

        processed_texts_an = self.process_list_text(anatomical_parts)
        inputs_an = self.tokenizer(processed_texts_an, add_special_tokens=True, max_length=256, padding='longest',
                                truncation=True, return_tensors='pt')
        inputs_an = {key: val.to(self.device) for key, val in inputs_an.items()}
        me_input_ids, attention_mask = inputs_an['input_ids'], inputs_an['attention_mask']
        _, text_features_cross_an, _ = self.text_encoder({'input_ids': me_input_ids, 'attention_mask': attention_mask})


        vision_features_itm_an = self.cross_encoder(vision_features_cross, text_features_cross_an).mean(dim=1)
        text_features_itm_an = self.cross_encoder(text_features_cross_an, vision_features_cross).mean(dim=1)

        processed_texts_me = self.process_list_text(medical_terms)
        inputs = self.tokenizer(processed_texts_me, add_special_tokens=True, max_length=256, padding='longest',
                                truncation=True, return_tensors='pt')
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        me_input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        _, text_features_cross, _ = self.text_encoder({'input_ids': me_input_ids, 'attention_mask': attention_mask})
        print(vision_features_cross.shape, text_features_cross.shape)
        vision_features_itm_me = self.cross_encoder(vision_features_cross, text_features_cross).mean(dim=1)
        text_features_itm_me = self.cross_encoder(text_features_cross, vision_features_cross).mean(dim=1)

        masked_image = self.mask_image_patches(images_1)
        features_1, features_2, _ = self.vision_encoder(masked_image)
        features_1 = self.cross_encoder(features_1, text_features_cross)
        reconstructed_image = self.MAEdecoder(features_1)

        return vision_features_clip, text_features_clip, vision_features_itm_an, text_features_itm_an, vision_features_itm_me, text_features_itm_me, reconstructed_image, images_1, text_features_MLM, text_labels


