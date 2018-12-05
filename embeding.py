import numpy as np
import os
from preprocessing.tokenization import get_words_feature
base=os.path.dirname(__file__)
def embeding_matrix():
    other=np.zeros(300)
    other=np.reshape(other,[1,300])
    with open(base+'/model_trained/w2v.txt','r',encoding='utf-8') as file:
        text=file.read()
        text=text.splitlines()
        matrix=[]
        for item in text:
            matrix.append(item.split())
        matrix=np.asarray(matrix)
        label=matrix[0:,0]
        label=np.concatenate((label,['other']),axis=0)
        # print(label[2600,])
        label=np.asarray(label)
        matrix=np.delete(matrix,0,1)
        matrix=np.concatenate((matrix,other),axis=0)
        # print(matrix[2600])
        file.close()
        # print(label.shape)
    return label,matrix
def embeding(document,padding):
    pos=[]
    embeder=[]
    label,matrix=embeding_matrix()
    label=list(label)
    list_word=get_words_feature(document)
    for word in list_word:
        if word in label:
            pos.append(label.index(word))
        else:pos.append(len(label)-1)
    for index in pos:
        embeder.append(matrix[index,0:])
    embeder=np.asarray(embeder)
    embeder=np.delete(embeder,np.arange(padding,embeder.shape[0]),0)
    embeder=np.concatenate((embeder,np.zeros((padding-embeder.shape[0],embeder.shape[1]))),axis=0)
    return embeder