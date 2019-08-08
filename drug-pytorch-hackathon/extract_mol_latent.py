import numpy as np
from jtnn import *
import rdkit
import torch

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
	print(gen_latent(np.loadtxt("test.txt", dtype="str")))



