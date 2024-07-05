import os
import pandas as pd

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from util.utils import *
from rdkit import Chem
from tqdm import tqdm
from train import markerModel
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

device_count = torch.cuda.device_count()
device_biomarker = torch.device('cuda' if torch.cuda.is_available() else "cpu")

device = torch.device('cpu')
d_model_name = 'DeepChem/ChemBERTa-10M-MTR'
p_model_name = 'DeepChem/ChemBERTa-10M-MLM'

tokenizer = AutoTokenizer.from_pretrained(d_model_name)
prot_tokenizer = AutoTokenizer.from_pretrained(p_model_name)

#--biomarker Model
##-- hyper param config file Load --##
config = load_hparams('config/predict.json')
config = DictX(config)
model = markerModel.load_from_checkpoint(config.load_checkpoint,strict=False)
        
# model = BiomarkerModel.load_from_checkpoint('./biomarker_bindingdb_train8595_pretopre/3477h3wf/checkpoints/epoch=30-step=7284.ckpt').to(device_biomarker)

model.eval()
model.freeze()
    
if device_biomarker.type == 'cuda':
    model = torch.nn.DataParallel(model)
    
def get_biomarker(drug_inputs, prot_inputs):
    output_preds = model(drug_inputs, prot_inputs)
    
    predict = torch.squeeze((output_preds)).tolist()

    # output_preds = torch.relu(output_preds)
    # predict = torch.tanh(output_preds)
    # predict = predict.squeeze(dim=1).tolist()

    return predict


def biomarker_prediction(smile_acc, smile_don):
    try:
        aas_input = smile_acc
       
            
        das_input =smile_don
        d_inputs = tokenizer(aas_input, padding='max_length', max_length=510, truncation=True, return_tensors="pt")
        # d_inputs = tokenizer(smiles, truncation=True, return_tensors="pt")
        drug_input_ids = d_inputs['input_ids'].to(device)
        drug_attention_mask = d_inputs['attention_mask'].to(device)
        drug_inputs = {'input_ids': drug_input_ids, 'attention_mask': drug_attention_mask}

        p_inputs = prot_tokenizer(das_input, padding='max_length', max_length=510, truncation=True, return_tensors="pt")
        # p_inputs = prot_tokenizer(aas_input, truncation=True, return_tensors="pt")
        prot_input_ids = p_inputs['input_ids'].to(device)
        prot_attention_mask = p_inputs['attention_mask'].to(device)
        prot_inputs = {'input_ids': prot_input_ids, 'attention_mask': prot_attention_mask}

        output_predict = get_biomarker(drug_inputs, prot_inputs)
        
        return output_predict

    except Exception as e:
        print(e)
        return {'Error_message': e}


def smiles_adp_test(smile_acc,smile_don):
    mola =  Chem.MolFromSmiles(smile_acc) 
    smile_acc = Chem.MolToSmiles(mola,   canonical=True)
    mold =  Chem.MolFromSmiles(smile_don)  
    smile_don = Chem.MolToSmiles(mold, canonical=True)

    batch_size = 1
    try:
        output_pred = biomarker_prediction((smile_acc), (smile_don))

        datas = output_pred

        ## -- Export result data to csv -- ##
        # df = pd.DataFrame(datas)
        # df.to_csv('./results/predict_test.csv', index=None)

        # print(df)
        return datas
        
    except Exception as e:
        print(e)
        return {'Error_message': e}
    

if __name__ == "__main__":
    a = smiles_adp_test('CCCCC(CC)CC1=C(F)C=C(C2=C3C=C(C4=CC=C(C5=C6C(=O)C7=C(CC(CC)CCCC)SC(CC(CC)CCCC)=C7C(=O)C6=C(C6=CC=C(C)S6)S5)S4)SC3=C(C3=CC(F)=C(CC(CC)CCCC)S3)C3=C2SC(C)=C3)S1','CCCCC(CC)CC1=CC=C(C2=C3C=C(C)SC3=C(C3=CC=C(CC(CC)CCCC)S3)C3=C2SC(C2=CC4=C(C5=CC(Cl)=C(CC(CC)CCCC)S5)C5=C(C=C(C)S5)C(C5=CC(Cl)=C(CC(CC)CCCC)S5)=C4S2)=C3)S1')                         
    
                     
    
