import pandas as pd
import torch
import os
from datetime import datetime as dt
from tqdm import tqdm
# If on the remote server
# from kgen2.mutual_information.end_to_end_analysis import analyzer
# from kgen2.mutual_information.fastGraph import fastGraphWithAnalyzer as FGA
# from kgen2.mutual_information.entropy import entropy_cdf
# from kgen2.LM.LM.RoBERTa import RoBERTa
from kgen2.CEDA import ceda_model



###########################################################################################
###### Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

start = dt.now()
PATH = '/home/zprosen/d/Grace2024/'
output_name = os.path.join(PATH,'ckpts','{}-recplot.pt')
RAW_DATA_PATH = os.path.join(PATH, 'data')
dataset = [os.path.join(RAW_DATA_PATH,f) for f in os.listdir(RAW_DATA_PATH) if not f.startswith('._')]

print(PATH, '\n\n')

level = [7, -1]

###########################################################################################
###### Process
###########################################################################################

for f in dataset:
    output_name_ = str(output_name).format(f.split('/')[-1].replace('.csv', ''))

    df = pd.read_csv(f)
    df = df.loc[
        ~df['text'].isna()
        & ~df['text2'].isna()
    ]

    print(df['speaker'].unique(), df['speaker2'].unique())
    # print(df.isna().sum())

    meta_data_cols = list(df)#[col for col in list(df) if 'text' not in col]

    GRAPH = ceda_model(
        sigma=1.5,
        device='cuda',
        wv_model='roberta-base',
        wv_layers=level
    )

    GRAPH.fit(df['text'].values.tolist(), df['text2'].values.tolist())

    # try:
    #     GRAPH.checkpoint(output_name)
    # except Exception:
    #     0

    GRAPH.meta_data = df[meta_data_cols].to_dict(orient='records')
    GRAPH.checkpoint(output_name_)

print('=======][=======\n')
