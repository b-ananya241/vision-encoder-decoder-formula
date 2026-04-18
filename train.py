import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
from pathlib import Path
from PIL import Image

from tokenizers import ByteLevelBPETokenizer
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from torch.utils.data import Dataset

# ================= CONFIG =================
TRAIN_BATCH_SIZE = 20
VALID_BATCH_SIZE = 5
LEARNING_RATE = 1e-4
TRAIN_EPOCHS = 4
WEIGHT_DECAY = 0.01
MAX_LEN = 128

data_dir = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DATA =================
def read_data(csv_file_path, typed='hw', split='train'):
    df = pd.read_csv(csv_file_path)
    image_formula_dict = df.set_index('image')['formula'].to_dict()

    if typed == 'hw':
        if split == 'train':
            images_path = data_dir + '/col_774_A4_2023/HandwrittenData/images/train/'
        else:
            images_path = data_dir + '/col_774_A4_2023/HandwrittenData/images/test/'
    else:
        images_path = data_dir + '/col_774_A4_2023/SyntheticData/images/'

    updated = {}
    for img, formula in image_formula_dict.items():
        if img.endswith('png'):
            new_path = images_path + img.split('/')[-1]
            updated[new_path] = '<s>' + formula + '<e>'

    df = pd.DataFrame({'image': list(updated.keys()), 'formula': list(updated.values())})
    return updated, df


# ================= TOKENIZER =================
def train_tokenizer(df):
    os.makedirs("tokenizer", exist_ok=True)

    temp_dir = "tokenizer/tmp"
    os.makedirs(temp_dir, exist_ok=True)

    # write formulas to text files
    for i, row in enumerate(df["formula"]):
        with open(f"{temp_dir}/{i}.txt", "w", encoding="utf-8") as f:
            f.write(row)

    paths = [str(x) for x in Path(temp_dir).glob("*.txt")]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=paths,
        vocab_size=10000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "<e>", "<unk>", "<mask>"]
    )

    tokenizer.save_model("tokenizer")
    return RobertaTokenizerFast.from_pretrained("tokenizer", max_len=MAX_LEN)


# ================= DATASETS =================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.examples = [
            tokenizer.encode_plus(t, max_length=MAX_LEN, truncation=True, padding="max_length")["input_ids"]
            for t in texts
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])


class ImageTextDataset(Dataset):
    def __init__(self, df, tokenizer, feature_extractor, max_length=500):
        self.df = df
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.image.iloc[idx]
        caption = self.df.formula.iloc[idx]

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.squeeze()

        labels = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        ).input_ids

        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


# ================= MAIN =================
if __name__ == "__main__":

    # Load data
    _, df_train_hw = read_data(data_dir + "/col_774_A4_2023/HandwrittenData/train_hw.csv")
    _, df_val_hw = read_data(data_dir + "/col_774_A4_2023/HandwrittenData/val_hw.csv")
    _, df_train = read_data(data_dir + "/col_774_A4_2023/SyntheticData/train.csv", typed='syn')
    _, df_val = read_data(data_dir + "/col_774_A4_2023/SyntheticData/val.csv", typed='syn')

    df_combined = pd.concat([df_train_hw, df_val_hw, df_train, df_val])

    # Train tokenizer
    tokenizer = train_tokenizer(df_combined)

    # ================= MLM TRAINING =================
    config = RobertaConfig(
        vocab_size=10000,
        num_attention_heads=12,
        num_hidden_layers=6
    )

    model = RobertaForMaskedLM(config=config).to(device)

    train_dataset = TextDataset(df_combined['formula'], tokenizer)
    eval_dataset = TextDataset(df_val_hw['formula'], tokenizer)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    training_args = TrainingArguments(
        output_dir="roberta_mlm",
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model("roberta_mlm")

    # ================= VISION ENCODER-DECODER =================
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k",
        "roberta_mlm"
    ).to(device)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = ImageTextDataset(df_combined, tokenizer, feature_extractor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="vit_roberta",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        fp16=True,
        save_total_limit=1
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator
    )

    trainer.train()
    trainer.save_model("vit_roberta")

    print("Training complete.")
