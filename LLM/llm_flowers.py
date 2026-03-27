from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain

import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from scipy.stats import pearsonr
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

llama_model_path = ''
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


prompt = """
    Based on the context, describe the prettiest flower in a single sentence that starts with "The prettiest flower is"
    CONTEXT: {}
    ANSWER:
    """

n = 30
T = 80
s = 1

data = np.load('RAGs.npz')['arr_0']

rag_embed = []
for i in range(n):
    rag_i = []
    for j in range(len(data[0][0])):
        rag_i.append(embedding_model.encode(data[i][0][j]))
    rag_embed.append(rag_i)
    
query_embed = embedding_model.encode('Describe the prettiest flower in a single sentence.')

retrieve_pos = []
for i in range(n):
    dotprod = []
    for j in range(len(data[0][0])):
        dotprod.append(np.sum(rag_embed[i][j]*query_embed))
    retrieve_pos.append(np.argmax(dotprod))  

print(retrieve_pos)


for p in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    for k in [1,3,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29]:
        for v in range(8):
            llm = LlamaCpp(model_path=llama_model_path,temperature=0.75,max_tokens=500,top_p=1
                           ,seed=11+v,n_ctx=2048,use_mlock=True,n_gpu_layers=40,n_threads=8,n_batch=1000
                           ,callback_manager=callback_manager,verbose=True)
            np.random.seed(2024)

            OUT = []; AUG = []; X = []; NB = []
            data = np.load('RAGs.npz')['arr_0']

            for t in range(T):
                mirrors = np.random.binomial(1,p,n)

                X_t = []; out_t = []
                for i in range(n):
                    X_it = []; out_it = []
                    for j in range(s):
                        updated_prompt = prompt.format(data[i][1][retrieve_pos[i]])
                        out = llm(updated_prompt)
                        count = 0
                        while ('The prettiest flower' not in out) and (count<4):
                            out = llm(updated_prompt)
                            count = count + 1
                        out_it.append(out)
                        embedding = embedding_model.encode(out)
                        X_it.append(embedding)
                    X_t.append(X_it); out_t.append(out_it)
                OUT.append(out_t)
                X.append(X_t)
    
                # Compute the distance matrix at time t.
                D_t = np.zeros((n,n))
                for i in range(n):
                    for j in range(i+1,n):
                        D_t[i][j] = np.linalg.norm(np.array(X_t[i])-np.array(X_t[j]))
                        D_t[j][i] = D_t[i][j]
            
                aug_t = []; nb_t = []
                # Update the augmentation.
                for i in range(n):
                    if mirrors[i]==1:
                        data[i][1][retrieve_pos[i]] = out_t[i][0]
                        aug_t.append(data[i][1][retrieve_pos[i]])
                        nb_t.append(i)
                    else:
                        sort = list(D_t[i].argsort())
                        sort.remove(i)
                        j = random.choice(sort[0:k])
                        nb_t.append(j)

                        data[i][1][retrieve_pos[i]] = out_t[j][0]
                        aug_t.append(data[i][1][retrieve_pos[i]])
    
                AUG.append(aug_t)
                NB.append(nb_t)
                print(t)
                print('----------------')

            np.savez('X'+str(int(p*10))+'_'+str(k)+'_'+str(v)+'.npz',np.array(X))
            pd.DataFrame(OUT).to_csv('output'+str(int(p*10))+'_'+str(k)+'_'+str(v)+'.csv')
            pd.DataFrame(AUG).to_csv('aug'+str(int(p*10))+'_'+str(k)+'_'+str(v)+'.csv')
            pd.DataFrame(NB).to_csv('nb'+str(int(p*10))+'_'+str(k)+'_'+str(v)+'.csv')
            print(p,k,v,'!!!!!!!!!!!!!!!!')
