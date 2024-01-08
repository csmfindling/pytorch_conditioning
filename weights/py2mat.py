import torch
from scipy.io import savemat, loadmat
from glob import glob

paths = glob('noise_level_0_5_entropy_level_-1_simul_id_*')
for p in paths:
    w = {k: v.detach().numpy() for (k, v) in torch.load(p).items()}
    savemat(f"weights_mat/{p}.mat", dict(w))

loaded_w = loadmat(f"weights_mat/{p}.mat")
