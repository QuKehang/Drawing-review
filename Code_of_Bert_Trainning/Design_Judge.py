#!/usr/bin/env python3
"""
BERT 文本分类模型 —— 设计说明合规性判别
基于 bert-base-chinese 微调二分类（符合规范 / 不符合规范）
"""

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_SAVE_PATH = "./bert_model_trained"

# 数据文件路径（与脚本同目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DESIGN_SPEC_FILE = os.path.join(SCRIPT_DIR, "design_spec.txt")
RELATION_NO_FILE = os.path.join(SCRIPT_DIR, "relation_no.txt")


# ═══════════════════════════════════════════════════════════
#  数据集准备
# ═══════════════════════════════════════════════════════════

def load_texts(file_path: str) -> list[str]:
    """加载文本文件，返回去除空白后的文本行列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    merged = "".join(line.strip() for line in lines)
    merged = merged.replace(" ", "")
    return [merged]


def build_dataset():
    """加载数据并构建标签"""
    # 符合规范的文本（正例）
    design_specs = load_texts(DESIGN_SPEC_FILE)
    labels_pos = [1] * len(design_specs)

    # 不符合规范的文本（负例）
    relations_no = load_texts(RELATION_NO_FILE)
    labels_neg = [0] * len(relations_no)

    # 合并
    all_texts = design_specs + relations_no
    all_labels = labels_pos + labels_neg

    print(f"正例（符合规范）: {len(design_specs)} 篇")
    print(f"负例（不符合规范）: {len(relations_no)} 篇")
    print(f"总计: {len(all_texts)} 篇")
    return all_texts, all_labels


# ═══════════════════════════════════════════════════════════
#  数据集类
# ═══════════════════════════════════════════════════════════

class DesignSpecDataset(Dataset):
    """设计说明数据集 —— 按句号拆分句子，每句继承文本的标签"""

    def __init__(self, tokenizer, texts, labels, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._build_examples(texts, labels)

    def _build_examples(self, texts, labels):
        examples = []
        for text, label in zip(texts, labels):
            sentences = text.split("。")
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    examples.append((sentence, label))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════
#  训练
# ═══════════════════════════════════════════════════════════

def train(model, data_loader, optimizer, criterion, device, epochs=EPOCHS):
    """训练循环"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs} — 平均训练损失: {avg_loss:.4f}")

    # 保存模型
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"模型已保存至: {MODEL_SAVE_PATH}")


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    print(f"设备: {DEVICE}")
    print("-" * 50)

    # 1. 加载数据
    print("[1/4] 加载数据集...")
    texts, labels = build_dataset()

    # 2. 初始化 tokenizer 和 DataLoader
    print("[2/4] 初始化 Tokenizer 和 DataLoader...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    dataset = DesignSpecDataset(tokenizer, texts, labels, MAX_LENGTH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  句子总数: {len(dataset)}")

    # 3. 初始化模型
    print("[3/4] 初始化 BERT 分类模型...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", num_labels=2
    )
    model.to(DEVICE)

    # 4. 训练
    print("[4/4] 开始训练...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, data_loader, optimizer, criterion, DEVICE)

    print("-" * 50)
    print("训练完成！")


if __name__ == "__main__":
    main()
