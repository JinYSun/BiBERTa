# **DeepDAP**

### **<u>Deep learning-assisted to accelerate the discovery of donor/acceptor pairs for high-performance organic solar cells</u>**

![overview](DeepDAP\overview.jpg)

## <u>Motivation</u>

It is a deep learning-based framework built for new donor/acceptor pairs (DeepDAP) discovery. The framework contains data collection section, PCE prediction section and molecular discovery section. Specifically, a large D/A pair dataset was built by collecting experimental data from literature. Then, a novel RoBERTa-based dual-encoder model (DeRoBERTa) was developed for PCE prediction by using the SMILES of donor and acceptor pairs as the input. Two pretrained ChemBERTa2 encoders were loaded as initial parameters of the dual-encoder. The model was trained, tested and validated on the experimental dataset.

## <u>Depends</u>

We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).

torch==1.10.0+cu113, 

pytorch-lightning==1.9.2, 

transformers==4.12.0,

numpy==1.20.0, 

pandas==1.4.3, 

curses=2.2.1+utf8,

scikit-learn==1.1.2,

scipy==1.4.1,

tqdm==4.66.1,

easydict==1.10

plotly==5.3.1

## <u>Usage</u>

-- train:
    contains the codes for training the model.
    
-- predict:
    contain the code for screening large-scale DAPs.
    
-- run:
    contain the code to predict the performance of DAP one by one. 

-- dataset/OSC

​	contain the dataset for training/testing/validating the model.

## <u>Discussion</u> 

The ***Discussion*** folder contains the scripts for evaluating the PCE prediction performance.  We compared sevaral methods widely used in molecular property prediction.



## <u>Contact</u>

Jinyu Sun. E-mail: [jinyusun@csu.edu.cn](mailto:jinyusun@csu.edu.cn)
