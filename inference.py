import pandas as pd
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, RobertaTokenizerFast
from PIL import Image
import torch
import sys

data_dir = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("vit_roberta").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = RobertaTokenizerFast.from_pretrained("tokenizer")

df = pd.read_csv(data_dir + "/sample_sub.csv")

predictions = []

for img_path in df["image"]:
    image = Image.open(img_path).convert("RGB")
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)

    output = model.generate(pixel_values, max_length=500)
    pred = tokenizer.decode(output[0])

    predictions.append(pred)

df["formula"] = predictions
df.to_csv("predictions.csv", index=False)

print("Done")
