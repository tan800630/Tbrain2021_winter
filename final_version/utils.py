import numpy as np
import pandas as pd
import pickle

def DCG(pred_file, ground_truth_pkl = 'm24_iDCG.pkl'):
    '''
    pred_file : csv file with columns : [chid, top1, top2, top3]
    ground_truth_pkl : pickle file for iDCG calculation at month k (default : 24).
    
    # note : only 398,988 users have transactions on month 24 with history before month 24, so we only compute those users' DCG score.    
    '''
    
    pred_class = {'2', '6', '10', '12', '13', '15', '18', '19', '21', '22', '25', '26', '36', '37', '39', '48'}
    
    # assert top3 shop_tag on pred_file in 16 pred_class
    assert len(set(np.unique(pred_file.values[:,1:])).difference(pred_class))==0
    
    with open(ground_truth_pkl, 'rb') as f:
        gd_dict, gd_df = pickle.load(f)
    
    print('Calculate DCG score with {:.0f} users'.format(np.sum(pred_file['chid'].isin(gd_dict.keys()))))
    
    with open(ground_truth_pkl, 'rb') as f:
        gd_dict, gd_df = pickle.load(f)

    pred_file = pd.merge(pred_file, gd_df[['chid', 'shop_tag', 'txn_amt']], left_on=['chid', 'top1'], right_on=['chid', 'shop_tag'], how='left')
    pred_file.drop(columns='shop_tag', inplace=True)
    pred_file.rename(columns={'txn_amt': 'top1_amt'}, inplace=True)

    pred_file = pd.merge(pred_file, gd_df[['chid', 'shop_tag', 'txn_amt']], left_on=['chid', 'top2'], right_on=['chid', 'shop_tag'], how='left')
    pred_file.drop(columns='shop_tag', inplace=True)
    pred_file.rename(columns={'txn_amt': 'top2_amt'}, inplace=True)

    pred_file = pd.merge(pred_file, gd_df[['chid', 'shop_tag', 'txn_amt']], left_on=['chid', 'top3'], right_on=['chid', 'shop_tag'], how='left')
    pred_file.drop(columns='shop_tag', inplace=True)
    pred_file.rename(columns={'txn_amt': 'top3_amt'}, inplace=True)

    pred_file = pred_file.fillna(0)

    iDCG = pd.Series(gd_dict).reset_index().rename(columns={'index': 'chid', 0: 'iDCG'})
    pred_file = pd.merge(pred_file[['chid', 'top1_amt', 'top2_amt', 'top3_amt']], iDCG, on='chid')
    pred_file['NDCG'] = pred_file.apply(lambda x: (x['top1_amt']/np.log2(2) + x['top2_amt']/np.log2(3) + x['top3_amt']/np.log2(4)) / x['iDCG'], axis=1)

    return pred_file.NDCG.mean()
