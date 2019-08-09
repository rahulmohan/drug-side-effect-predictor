import numpy as np
from jtnn import *
import rdkit
import torch
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

def load_SIDER():
	sider = pd.read_csv("SIDER_PTs.csv")
	meta_smiles = pd.read_csv("meta_SMILES.csv", index_col="pert_id")
	sider_smiles = meta_smiles.loc[sider.values[:,0]]["SMILES"].values
	out_file = open("sider_smiles.txt", "w")
	for smile in sider_smiles:
		out_file.write(smile + "\n")
	out_file.close()

def gen_latent_SIDER():
	smiles = np.loadtxt("sider_smiles.txt", dtype="str")
	latent_all = []
	idxs_to_keep = []
	for i in range(len(smiles)):
		try:
			latent = gen_latent([smiles[i]])
			latent_all.append(latent.flatten())
			idxs_to_keep.append(i)
		except:
			continue
	np.save("sider_latent.npy", np.array(latent_all))
	np.save("sider_idxs_to_keep.npy", np.array(idxs_to_keep))

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def train_sider_model():
	latent = np.load("sider_latent.npy")
	idxs_to_keep = np.load("sider_idxs_to_keep.npy")
	sider = pd.read_csv("SIDER_PTs.csv").values
	sider_labels = sider[idxs_to_keep, 1:].astype('int')
	X_train, X_test, y_train, y_test = train_test_split(latent, sider_labels, test_size=0.2)
	classif = OneVsRestClassifier(KNeighborsClassifier())
	classif.fit(X_train, y_train)
	preds = classif.predict(X_test)
	print(f1_score(y_test, preds, average="micro"))
	print(precision_score(y_test, preds, average="micro"))
	print(recall_score(y_test, preds, average="micro"))

def gen_latent(smiles):
	# Model parameters
	hidden_size = int(450)
	latent_size = int(56)
	depth = 3
	stereo = True
	vocab = [x.strip("\r\n ") for x in open("vocab.txt")] 
	vocab = Vocab(vocab)
	# Load model weights
	model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
	load_dict = torch.load("MPNVAE-h450-L56-d3-beta0.001/model.iter-4", map_location=torch.device('cpu'))
	missing = {k: v for k, v in model.state_dict().items() if k not in load_dict}
	load_dict.update(missing) 
	model.load_state_dict(load_dict)
	# Extract latent representation
	mol_vec = model.encode_latent_mean(smiles)
	mol_vec = mol_vec.data.cpu().numpy()
	return mol_vec


if __name__ == '__main__':
	#print(gen_latent(np.loadtxt("sider_smiles.txt", dtype="str"), "vocab.txt"))
	train_sider_model()


