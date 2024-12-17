import numpy as np
import scipy
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, roc_auc_score
for prompt in ['J', 'A','E','F','H']:
    print (prompt)
    auc1=[]
    for seed in range(5):
    
        human=np.load("{}/seed{}_human_val.npy".format(prompt, seed))
        llamaLogits=np.load("{}/seed{}_llama_logits_val.npy".format(prompt, seed))
        llama=np.argmax(llamaLogits, axis=1)
        #print(np.mean(llama==human))
        #print(balanced_accuracy_score(human, llama))
        #print(roc_auc_score(human, llamaLogits[:,1]))
        #auc1.append(roc_auc_score(human, llamaLogits[:,1]))
        probs=scipy.special.softmax(llamaLogits, axis=1)
        fpr, tpr, threshold = roc_curve(human, probs[:, 1])
        
        auc1.append(auc(fpr, tpr))
    print(auc1)
    #print(np.mean(roc_auc))
    #print(np.std(roc_auc))
    print(np.mean(auc1))
    print(np.std(auc1))

