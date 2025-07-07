import torch
import os
import torch.nn as nn
from transformers import AutoTokenizer
from Model.Image_encoder import SwinTransformer
from Model.Text_encoder import text_encoder
from torchvision import transforms
from PIL import Image
from Model.Cross_Modal_Fusion import MultiLayerEncoder
from Model.Generation.Transformer_Decoder import HKRGDecoder
torch.autograd.set_detect_anomaly(True)


class HKRGModel_G(nn.Module):
    """
    text_model_path : Load  tokenizer
    pretrained_model_path : Load  pre-training model weight
    """
    def __init__(self,text_model_path=r"E:\Medical image report generation\model\MedBert",
                 pretrained_model_path=r"E:\Medical image report generation\best_model.pth"):
        super(HKRGModel_G, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize all model components
        self.vision_encoder = SwinTransformer().to(self.device)
        self.text_encoder = text_encoder(text_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.vocab_size = len(self.tokenizer)
        self.cross_encoder=MultiLayerEncoder(768, 8, 6).to(self.device)
        self.cross_conv = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, padding=1)
        self.self_attn = nn.MultiheadAttention(768, 8).to(self.device)
        self.adaptive_pool_text = nn.AdaptiveAvgPool2d((49, 768))
        self.norm = nn.LayerNorm(768)
        self.decoder = HKRGDecoder(d_model=768, nhead=8, num_layers=12, vocab_size=self.vocab_size).to(self.device)

        # Try to load the pretrained model
        if os.path.exists(pretrained_model_path):
            try:
                checkpoint = torch.load(pretrained_model_path, map_location=self.device)
                missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
                if missing_keys:
                    print(f"Missing keys in state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in state_dict: {unexpected_keys}")
                print("Pretrained model loaded successfully.")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}. Continuing with default initialization.")
        else:
            print("No pretrained model file found at specified path. Continuing with default initialization.")

    def process_image(self, image_paths):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        images = [transform(Image.open(img_path).convert('RGB')) for img_path in image_paths]
        images = torch.stack(images)
        return images.to(self.device)

    def process_text(self, texts):

        encoded_texts = [self.tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
                         for text in texts]
        max_len = max(len(x) for x in encoded_texts)
        padded_texts = [text + [self.tokenizer.pad_token_id] * (max_len - len(text)) for text in encoded_texts]
        padded_texts_tensor = torch.tensor(padded_texts, dtype=torch.long)

        return padded_texts_tensor.to(self.device)

    def process_list_text(self, texts_batch):
        processed_texts = []
        for sublist in texts_batch:
            text_token = self.tokenizer(sublist, return_tensors="pt", padding=True, truncation=True, max_length=512)
            processed_texts.append({key: val.to(self.device) for key, val in text_token.items()})
        return processed_texts

    def decode_text(self, tensor):
        texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in tensor]
        return texts

    def image_Cross_fusion(self, features_cross_1, features_cross_2):
        concatenated_tensor = torch.cat((features_cross_1, features_cross_2), dim=1).permute(0, 2, 1)
        vision_features_cross = self.cross_conv(concatenated_tensor)
        vision_features_cross = vision_features_cross.permute(0, 2, 1)
        return vision_features_cross

    def forward(self, images_1, images_2, reports, anatomical_parts, medical_terms):

        # Image processing for images_1
        images_1 = self.process_image(images_1)
        if images_1 is None:
            raise ValueError("Image processing failed.")
        vision_features_cross_1, vision_features_clip_1, _ = self.vision_encoder(images_1)

        # Image processing for images_2
        images_2 = self.process_image(images_2)
        if images_2 is None:
            raise ValueError("Image processing failed.")
        vision_features_cross_2, vision_features_clip_2, _ = self.vision_encoder(images_2)

        # Concatenate features from image 1 and 2
        vision_features = self.image_Cross_fusion(vision_features_cross_1, vision_features_cross_2)

        # Encode anatomical part text
        processed_anatomical_parts = self.process_list_text(anatomical_parts)
        text_features_an = []
        for text_token_an in processed_anatomical_parts:
            text_feature_an_1, _, _ = self.text_encoder(text_token_an)
            text_features_an.append(text_feature_an_1)

        text_features_an = torch.stack(text_features_an, dim=0)
        text_vision_features_an = self.cross_encoder(text_features_an, vision_features)
        vision_text_features_an = self.cross_encoder(vision_features, text_features_an)

        combined_features = torch.cat((vision_text_features_an, text_vision_features_an), dim=1)
        combined_features_an = self.norm(combined_features)

        # Encode medical term text
        processed_Medical_parts = self.process_list_text(medical_terms)
        text_features_me = []
        for text_token_me in processed_Medical_parts:
            text_features_me_1, _, _ = self.text_encoder(text_token_me)
            text_features_me.append(text_features_me_1)
        text_features_me = torch.stack(text_features_me, dim=0)
        text_vision_features_me = self.cross_encoder(text_features_me, vision_text_features_an)
        vision_text_features_me = self.cross_encoder(vision_text_features_an, text_features_me)
        vision_text_features_me = vision_text_features_an + vision_text_features_me
        combined_features_me = torch.cat((vision_text_features_me, text_vision_features_me), dim=1)
        combined_features_me = self.norm(combined_features_me)

        #decoder
        texts=self.process_text(reports)
        output = self.decoder(texts, [vision_features, combined_features_an, combined_features_me])
        output = output.view(-1, output.size(-1))
        labels=texts.view(-1)
        return  output, labels

