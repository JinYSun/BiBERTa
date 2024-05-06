# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:06:46 2023

@author: BM109X32G-10GPU-02
"""
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import transformers
from transformers import BertForSequenceClassification
import rdkit
import transformers
import pandas as pd
import numpy as np
import sklearn
class CustomTrainDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
     
        item = df.iloc[idx]        
        text = "[CLS]" + item["donor"] + "[SEP]" + item["acceptor"] + "[SEP]"
        label = item['Label']       
        # encode text
        encoding = self.tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # add label
        encoding["labels"] = torch.tensor(label).type(torch.FloatTensor)
        return encoding

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
df = pd.read_csv(r"C:\Users\BM109X32G-10GPU-02\Desktop\crossdata\train.csv").iloc[:,1:]
train_dataset = CustomTrainDataset(df=df, tokenizer=tokenizer)
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# Instantiate pre-trained BERT model with randomly initialized classification head
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

class CustomvalDataset(Dataset):
    def __init__(self, df1, tokenizer):
        self.df1 = df1
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df1)

    def __getitem__(self, idx):
     
        item = df1.iloc[idx]        
        text = "[CLS]" + item["donor"] + "[SEP]" + item["acceptor"] + "[SEP]"
        label = item['Label']       
        # encode text
        encoding = self.tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # add label
        encoding["labels"] = torch.tensor(label).type(torch.FloatTensor)
        return encoding
df1 = pd.read_csv(r"C:\Users\BM109X32G-10GPU-02\Desktop\crossdata\test.csv").iloc[:,1:]
eval_dataset = CustomvalDataset(df1=df1, tokenizer=tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=True)

bert_model_name = "bert-base-uncased" # 使用英文版的BERT模型
input_data_path = "data/smiles.csv" # 输入数据的路径，假设是一个csv文件，包含三列分别是ligand_smiles、receptor_smiles和property
output_model_path = "model/bert_smiles.pt" # 输出模型的路径
batch_size = 10 # 批处理大小
learning_rate = 1e-4 # 学习率
num_epochs = 20 # 训练轮数
target_property = "Label" # 目标性质


# I almost always use a learning rate of 5e-5 when fine-tuning Transformer based models
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
loss_fn = torch.nn.MSELoss()
# put model on GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
trainer = transformers.Trainer(
  model=model, # 模型
  args=transformers.TrainingArguments( # 训练参数
    output_dir=output_model_path, # 输出路径
    num_train_epochs=num_epochs, # 训练轮数
    per_device_train_batch_size=batch_size, # 每个设备的批处理大小
    save_strategy="epoch",
    evaluation_strategy="epoch", # 每轮结束时评估模型
    load_best_model_at_end=True, # 训练结束时加载最佳模型
  ),
  train_dataset = train_dataset,
  eval_dataset = eval_dataset,
  compute_metrics=lambda x: {"mse": loss_fn(torch.tensor(x.predictions.flatten()), torch.tensor(x.label_ids.flatten())), 
                              "r2": sklearn.metrics.r2_score(x.label_ids.flatten(), x.predictions.flatten())})

# 开始训练
trainer.train()
# 训练结束后，保存模型到输出路径
trainer.save_model("model/bert_smiles.pt")
dff = pd.read_csv(r"C:\Users\BM109X32G-10GPU-02\Desktop\crossdata\val.csv").iloc[:,1:]

class CustomTestDataset(Dataset):
    def __init__(self, dff, tokenizer):
        self.dff = dff
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dff)

    def __getitem__(self, idx):
     
        item = dff.iloc[idx]        
        text = "[CLS]" + item["donor"] + "[SEP]" + item["acceptor"] + "[SEP]"
        label = item['Label']       
        # encode text
        encoding = self.tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # add label
        encoding["labels"] = torch.tensor(label).type(torch.FloatTensor)
        return encoding

test_dataset = CustomTestDataset(dff=dff, tokenizer=tokenizer)
predt = trainer.predict(test_dataset)

from sklearn.metrics import median_absolute_error,r2_score, mean_absolute_error,mean_squared_error
x = pd.DataFrame(predt[1]).iloc[:,0]
y = pd.DataFrame(predt[0]).iloc[:,0]
r2 = r2_score(x,y)

mae = mean_absolute_error(x,y)
mse = mean_squared_error(x,y)
   
from scipy.stats import pearsonr
print(pearsonr(x,y))
print(r2, mae , mse)
'''
for epoch in range(50):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        # put batch on device
        batch = {k:v.to(device) for k,v in batch.items()}
        # forward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss.type(torch.FloatTensor)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            # put batch on device
            batch = {k:v.to(device) for k,v in batch.items()}
            
            # forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            val_loss += loss.item()
                  
    print("Validation loss after epoch {epoch}:", val_loss/len(eval_dataloader))
'''