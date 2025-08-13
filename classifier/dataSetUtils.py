import os, glob, json
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

#Reprodutibilidade: fixe seed em train_test_split e também no PyTorch ao treinar.

#Estratificação: já aplicado; mantém proporção de classes.

#Balanceamento: se houver classes desbalanceadas, calcule class_weight (PyTorch: CrossEntropyLoss(weight=...)).

#Versione o .npz gerado; assim o treino fica independente do diretório de origem.

#def load_latents_and_labels(data_dir):
 #   X, y = [], []
  #  paths = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
   # for p in paths:
    #    d = np.load(p, allow_pickle=True)
     #   if 'z_txt' not in d or 'track_role' not in d:
      #      continue
      #  X.append(d['z_txt']) # (256,)
       # y.append(str(d['track_role'])) # string

    #X = np.stack(X, axis = 0)
    #y = np.array(y)

    #return X, y

def load_latents_and_labels(data_dir):
    X, y, ids = [], [], []
    paths = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if 'z_txt' not in d or 'track_role' not in d:
            continue
        X.append(d['z_txt'])                  # (256,)
        y.append(str(d['track_role']))        # string
        ids.append(os.path.basename(p))       # <-- NOVO: id = nome do arquivo

    X = np.stack(X, axis=0)
    y = np.array(y)
    ids = np.array(ids)

    return X, y, ids

def make_label_encoder(y):

    classes = sorted(np.unique(y).tolist())
    cls2id = {c:i for i,c in enumerate(classes)}
    y_id = np.array([cls2id[c] for c in y], dtype=np.int64)
    return y_id, classes, cls2id

def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0); sd[sd==0] = 1.0
    return mu, sd

def zscore_apply(X, mu, sd):
    return (X - mu) / sd

#def prepare_dataset(data_dir, out_npz="commu_z_txt_trackrole_dataset.npz",
#                    test_size=0.15, val_size=0.15, seed=42):

    # 1) carrega
 #   X, y = load_latents_and_labels(data_dir)

    # 2) codifica labels
 #   y_id, classes, cls2id = make_label_encoder(y)

    # 3) split estratificado: primeiro train+val vs test
 #   X_trv, X_te, y_trv, y_te = train_test_split(
 #       X, y_id, test_size=test_size, random_state=seed, stratify=y_id
 #   )
    # depois train vs val
 #   val_ratio = val_size / (1 - test_size)
 #   X_tr, X_va, y_tr, y_va = train_test_split(
 #       X_trv, y_trv, test_size=val_ratio, random_state=seed, stratify=y_trv
 #   )
    
    # 4) normaliza com estatísticas do treino
 #   mu, sd = zscore_fit(X_tr)
 #   X_tr = zscore_apply(X_tr, mu, sd)
 #   X_va = zscore_apply(X_va, mu, sd)
 #   X_te = zscore_apply(X_te, mu, sd)

    # 5) salva tudo num único arquivo
    #np.savez_compressed(
     #   out_npz,
      #  X_train=X_tr, y_train=y_tr,
       # X_val=X_va,  y_val=y_va,
       # X_test=X_te, y_test=y_te,
       # class_names=np.array(classes),
       # mu=mu, sd=sd)
    

    # salva mapeamento em JSON (opcional e útil)
 #   with open(out_npz.replace(".npz", "_label_map.json"), "w") as f:
#        json.dump({"class_to_id": cls2id, "classes": classes}, f, indent=2)

    # 6) sanity-check rápido
#    print("Shapes:", X_tr.shape, X_va.shape, X_te.shape)
 #   print("Classes:", classes)
#    return out_npz

#______________________________

def prepare_dataset(data_dir, out_npz="commu_z_txt_trackrole_dataset.npz",
                    test_size=0.15, val_size=0.15, seed=42):

    # 1) carrega
    X, y, ids = load_latents_and_labels(data_dir)

    # 2) codifica labels
    y_id, classes, cls2id = make_label_encoder(y)

    # 3) split estratificado: primeiro train+val vs test (inclui ids)
    X_trv, X_te, y_trv, y_te, ids_trv, ids_te = train_test_split(
        X, y_id, ids, test_size=test_size, random_state=seed, stratify=y_id
    )
    # depois train vs val (inclui ids)
    val_ratio = val_size / (1 - test_size)
    X_tr, X_va, y_tr, y_va, ids_tr, ids_va = train_test_split(
        X_trv, y_trv, ids_trv, test_size=val_ratio, random_state=seed, stratify=y_trv
    )
    
    # 4) normaliza com estatísticas do treino
    mu, sd = zscore_fit(X_tr)
    X_tr = zscore_apply(X_tr, mu, sd)
    X_va = zscore_apply(X_va, mu, sd)
    X_te = zscore_apply(X_te, mu, sd)

    # 5) salva tudo num único arquivo (inclui ids)
    np.savez_compressed(
        out_npz,
        X_train=X_tr, y_train=y_tr, ids_train=ids_tr,
        X_val=X_va,  y_val=y_va,  ids_val=ids_va,
        X_test=X_te, y_test=y_te, ids_test=ids_te,
        class_names=np.array(classes),
        mu=mu, sd=sd
    )

    # salva mapeamento em JSON (opcional e útil)
    with open(out_npz.replace(".npz", "_label_map.json"), "w") as f:
        json.dump({"class_to_id": cls2id, "classes": classes}, f, indent=2)

    # 6) sanity-check rápido
    print("Shapes:", X_tr.shape, X_va.shape, X_te.shape)
    print("Classes:", classes)
    return out_npz
#______________________________

#class LatentRoleDataset(Dataset):
 #   def __init__(self, X, y):
 #       self.X = torch.from_numpy(X).float()
 #       self.y = torch.from_numpy(y).long()

 #   def __len__(self): return self.X.shape[0]
 #   def __getitem__(self, i): return self.X[i], self.y[i]

#def load_npz_as_datasets(npz_path):
#    d = np.load(npz_path)
#    train_ds = LatentRoleDataset(d['X_train'], d['y_train'])
#    val_ds   = LatentRoleDataset(d["X_val"],   d["y_val"])
#    test_ds  = LatentRoleDataset(d["X_test"],  d["y_test"])
#    class_names = d["class_names"].tolist()
#    return train_ds, val_ds, test_ds, class_names

class LatentRoleDataset(Dataset):
    def __init__(self, X, y, ids=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        # ids é opcional; se vier, guarde para consulta externa
        self.ids = ids if ids is not None else np.array([str(i) for i in range(len(X))])

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, i): 
        # mantém compatibilidade: retorna (X, y)
        return self.X[i], self.y[i]

def load_npz_as_datasets(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    train_ds = LatentRoleDataset(d['X_train'], d['y_train'], d.get('ids_train'))
    val_ds   = LatentRoleDataset(d["X_val"],   d["y_val"],   d.get('ids_val'))
    test_ds  = LatentRoleDataset(d["X_test"],  d["y_test"],  d.get('ids_test'))
    class_names = d["class_names"].tolist()
    return train_ds, val_ds, test_ds, class_names
