import os
import pandas as pd

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from rdkit import Chem
from util.utils import *

from tqdm import tqdm
from train import bibertaModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0 '

device_count = torch.cuda.device_count()
device_biobiberta = torch.device('cuda' if torch.cuda.is_available() else "cpu")

device = torch.device('cpu')
a_model_name = 'DeepChem/ChemBERTa-10M-MLM'
d_model_name = 'DeepChem/ChemBERTa-10M-MTR'

tokenizer = AutoTokenizer.from_pretrained(a_model_name)
d_tokenizer = AutoTokenizer.from_pretrained(d_model_name)

#--biobiberta Model
##-- hyper param config file Load --##

config = load_hparams('config/predict.json')
config = DictX(config)
model = bibertaModel(config.d_model_name, config.p_model_name,
                              config.lr, config.dropout, config.layer_features, config.loss_fn, config.layer_limit, config.pretrained['chem'], config.pretrained['prot'])
#model.load_state_dict(torch.load(r"J:\libray\DeepDAP\DeepDAP\OSC\dap.pt"))
# model = BiobibertaModel.load_from_checkpoint('./biobiberta_bindingdb_train8595_pretopre/3477h3wf/checkpoints/epoch=30-step=7284.ckpt').to(device_biobiberta)
model = bibertaModel.load_from_checkpoint(config.load_checkpoint,strict=False)
model.eval()
model.freeze()
    
if device_biobiberta.type == 'cuda':
    model = torch.nn.DataParallel(model)
    
def get_biberta(drug_inputs, prot_inputs):
    output_preds = model(drug_inputs, prot_inputs)
   
    predict = torch.squeeze( (output_preds)).tolist()

    # output_preds = torch.relu(output_preds)
    # predict = torch.tanh(output_preds)
    # predict = predict.squeeze(dim=1).tolist()

    return predict


def biberta_prediction(smiles, aas):
    try:
        aas_input = []
        for ass_data in aas:
            aas_input.append(' '.join(list(ass_data)))
    
        a_inputs = tokenizer(smiles, padding='max_length', max_length=510, truncation=True, return_tensors="pt")
        # d_inputs = tokenizer(smiles, truncation=True, return_tensors="pt")
        a_input_ids = a_inputs['input_ids'].to(device)
        a_attention_mask = a_inputs['attention_mask'].to(device)
        a_inputs = {'input_ids': a_input_ids, 'attention_mask': a_attention_mask}

        d_inputs = d_tokenizer(aas_input, padding='max_length', max_length=510, truncation=True, return_tensors="pt")
        # p_inputs = prot_tokenizer(aas_input, truncation=True, return_tensors="pt")
        d_input_ids = d_inputs['input_ids'].to(device)
        d_attention_mask = d_inputs['attention_mask'].to(device)
        d_inputs = {'input_ids': d_input_ids, 'attention_mask': d_attention_mask}
 
        output_predict = get_biberta(a_inputs, d_inputs)
        
        output_list = [{'acceptor': smiles[i], 'donor': aas[i], 'predict': output_predict[i]} for i in range(0,len(aas))]

        return output_list

    except Exception as e:
        print(e)
        return {'Error_message': e}


def smiles_aas_test(file):
     
    batch_size = 80
    try:
        datas = []
        biberta_list = []
        biberta_datas = []

        smiles_aas = pd.read_csv(file)
        
        ## -- 1 to 1 pair predict check -- ##
        for data in smiles_aas.values:
            mola =  Chem.MolFromSmiles(data[2]) 
            data[2] = Chem.MolToSmiles(mola,   canonical=True)
            mola =  Chem.MolFromSmiles(data[1]) 
            data[1] = Chem.MolToSmiles(mola,   canonical=True)                        
            biberta_datas.append([data[2], data[1]])
            if len(biberta_datas) == batch_size:
                biberta_list.append(list(biberta_datas))
                biberta_datas.clear()

        if len(biberta_datas) != 0:
            biberta_list.append(list(biberta_datas))
            biberta_datas.clear()
            
        for biberta_datas in tqdm(biberta_list, total=len(biberta_list)):
            smiles_d , smiles_a  = zip(*biberta_datas)
            output_pred = biberta_prediction(list(smiles_d), list(smiles_a) )
            if len(datas) == 0:
                datas = output_pred
            else:
                datas = datas + output_pred

        # ## -- Export result data to csv -- ##
        # df = pd.DataFrame(datas)
        # df.to_csv('./results/predictData_nontonon_bindingdb_test.csv', index=None)

        # print(df)
        return datas
        
    except Exception as e:
        print(e)
        return {'Error_message': e}

