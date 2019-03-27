import pandas as pd

sub_attention = pd.read_csv('./cache/sub_Attention.csv',header=None)
sub_cnn = pd.read_csv('./cache/sub_CNN.csv',header=None)
sub_rnn = pd.read_csv('./cache/sub_RNN.csv',header=None)

sub_attention.columns = ['id','score']
sub_rnn.columns = ['id','score']
sub_cnn.columns = ['id','score']

sub = pd.DataFrame()
sub['id'] = sub_attention['id'].copy()

attenscore = sub_attention.score.values
attenscore[attenscore>4.7] = 5

rnnscore = sub_rnn.score.values
rnnscore[rnnscore>4.7] = 5

cnnscore = sub_cnn.score.values
cnnscore[cnnscore>4.7] = 5


score = (attenscore + rnnscore +cnnscore)/3
score[score>4.7] = 5

sub["score"] = pd.DataFrame(score)
print(sub)
sub.to_csv('./cache/final_sub.csv',index=False,header=False)
