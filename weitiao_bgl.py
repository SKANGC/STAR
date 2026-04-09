import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List
from peft import LoraConfig, get_peft_model

# 配置参数
class Config:
    # 数据参数
    csv_path = "./dbscan_test.csv"
    model_path = "./model"
    output_dir = "./fine_tuned_embedding_model"
    
    # 训练参数
    num_epochs = 10
    batch_size = 1
    learning_rate = 1e-5
    weight_decay = 0.01
    max_length = 512
    
    # 损失函数参数
    margin = 1.0
    temperature = 0.1
    
    # 其他参数
    log_interval = 50
    save_interval = 2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 数据集类
class LogEmbeddingDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=1024):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        required_columns = ['Content', 'EventTemplate', 'label', 'cluster']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"CSV文件必须包含'{col}'列")
        
        self.process_data()
        
    def process_data(self):
        self.samples = []
        
        cluster_groups = self.df.groupby('cluster')
        
        for cluster_id, group in cluster_groups:
            normal_samples = group[group['label'] == 0]
            abnormal_samples = group[group['label'] == 1]
            
            for _, sample in normal_samples.iterrows():
                positive_candidates = normal_samples[normal_samples.index != sample.name]
                if len(positive_candidates) > 0:
                    positive_sample = positive_candidates.sample(1).iloc[0]
                    negative_sample = self.get_negative_sample(sample, cluster_id)
                    
                    self.samples.append({
                        'anchor': sample,
                        'positive': positive_sample,
                        'negative': negative_sample,
                    })
            
            for _, sample in abnormal_samples.iterrows():
                positive_candidates = abnormal_samples[abnormal_samples.index != sample.name]
                if len(positive_candidates) > 0:
                    positive_sample = positive_candidates.sample(1).iloc[0]
                    negative_sample = self.get_negative_sample(sample, cluster_id)
                    
                    self.samples.append({
                        'anchor': sample,
                        'positive': positive_sample,
                        'negative': negative_sample,
                    })
    
    def get_negative_sample(self, anchor_sample, anchor_cluster):
        anchor_label = anchor_sample['label']
        
        different_cluster_samples = self.df[self.df['cluster'] != anchor_cluster]
        if len(different_cluster_samples) > 0:
            return different_cluster_samples.sample(1).iloc[0]
        
        same_cluster_diff_label = self.df[
            (self.df['cluster'] == anchor_cluster) & 
            (self.df['label'] != anchor_label)
        ]
        if len(same_cluster_diff_label) > 0:
            return same_cluster_diff_label.sample(1).iloc[0]
        
        other_samples = self.df[self.df.index != anchor_sample.name]
        if len(other_samples) > 0:
            return other_samples.sample(1).iloc[0]
        
        return anchor_sample
    
    def format_input_text(self, sample):
        system_prompt = (
            "你是一个安全分析助手，需要为进程信息生成联合向量表示，重点关注进程日志"
            "请综合分析以下3种信息：\n"
            "1. 进程运行时日志\n"
            "2. 运行环境\n"
        )
        
        user_input = (
            f"### 进程日志:\n{sample['Content']}\n\n"
            f"### 运行设备:\nBGL\n\n"
            "请综合分析上述信息生成嵌入表示"
        )
        
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        anchor_text = self.format_input_text(sample['anchor'])
        positive_text = self.format_input_text(sample['positive'])
        negative_text = self.format_input_text(sample['negative'])
        
        anchor_tokens = self.tokenizer(
            anchor_text, 
            return_tensors="pt", 
            padding='max_length',
            truncation=True, 
            max_length=self.max_length
        )
        
        positive_tokens = self.tokenizer(
            positive_text, 
            return_tensors="pt", 
            padding='max_length',
            truncation=True, 
            max_length=self.max_length
        )
        
        negative_tokens = self.tokenizer(
            negative_text, 
            return_tensors="pt", 
            padding='max_length',
            truncation=True, 
            max_length=self.max_length
        )
        
        return {
            'anchor_input_ids': anchor_tokens['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_tokens['attention_mask'].squeeze(),
            'positive_input_ids': positive_tokens['input_ids'].squeeze(),
            'positive_attention_mask': positive_tokens['attention_mask'].squeeze(),
            'negative_input_ids': negative_tokens['input_ids'].squeeze(),
            'negative_attention_mask': negative_tokens['attention_mask'].squeeze(),
        }

# 损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, anchor_emb, positive_emb, negative_emb):
        pos_similarity = F.cosine_similarity(anchor_emb, positive_emb)
        neg_similarity = F.cosine_similarity(anchor_emb, negative_emb)
        
        triplet_loss = F.relu(neg_similarity - pos_similarity + self.margin)
        
        similarities = torch.cat([pos_similarity.unsqueeze(1), neg_similarity.unsqueeze(1)], dim=1)
        contrastive_loss = F.cross_entropy(similarities / self.temperature, torch.zeros_like(pos_similarity, dtype=torch.long))
        
        total_loss = triplet_loss.mean() + contrastive_loss
        
        return total_loss

# 微调模型
class EmbeddingModelForFineTuning(nn.Module):
    def __init__(self, base_model, tokenizer):
        super(EmbeddingModelForFineTuning, self).__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.freeze_layers()
        
    def freeze_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        num_layers = len(self.base_model.model.layers)
        layers_to_unfreeze = min(4, num_layers)
        for i in range(num_layers - layers_to_unfreeze, num_layers):
            for param in self.base_model.model.layers[i].parameters():
                param.requires_grad = True
        
        for param in self.base_model.lm_head.parameters():
            param.requires_grad = True
    
    def get_embeddings(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states
        last_layers = hidden_states[-4:]
        stacked = torch.stack(last_layers)
        embeddings = stacked.mean(dim=0).mean(dim=1)
        
        return embeddings
    
    def forward(self, input_ids, attention_mask):
        return self.get_embeddings(input_ids, attention_mask)

# 训练函数
def train_embedding_model():
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_path,
        trust_remote_code=True,
        device_map=Config.device,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        Config.model_path,
        trust_remote_code=True
    )

    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    embedding_model = EmbeddingModelForFineTuning(model, tokenizer)
    embedding_model = embedding_model.to(model.device)
    
    dataset = LogEmbeddingDataset(
        csv_path=Config.csv_path,
        tokenizer=tokenizer,
        max_length=Config.max_length
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        shuffle=True
    )
    
    contrastive_loss = ContrastiveLoss(margin=Config.margin, temperature=Config.temperature)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, embedding_model.parameters()),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=Config.num_epochs * len(dataloader)
    )
    
    os.makedirs(Config.output_dir, exist_ok=True)
    
    embedding_model.train()
    best_loss = float('inf')
    
    for epoch in range(Config.num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            anchor_emb = embedding_model(
                batch['anchor_input_ids'],
                batch['anchor_attention_mask']
            )
            positive_emb = embedding_model(
                batch['positive_input_ids'], 
                batch['positive_attention_mask']
            )
            negative_emb = embedding_model(
                batch['negative_input_ids'],
                batch['negative_attention_mask']
            )
            
            loss = contrastive_loss(anchor_emb, positive_emb, negative_emb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % Config.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch {epoch+1}/{Config.num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_save_path = os.path.join(Config.output_dir, "best_model")
            embedding_model.base_model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
        
        if (epoch + 1) % Config.save_interval == 0:
            checkpoint_path = os.path.join(Config.output_dir, f"checkpoint_epoch_{epoch+1}")
            embedding_model.base_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
    
    final_model_path = os.path.join(Config.output_dir, "final_model")
    embedding_model.base_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    return embedding_model

if __name__ == "__main__":
    train_embedding_model()

