import numpy as np
import matplotlib.pyplot as plt
import torch 

# Carrega X e y de um dataset (val_ds ou test_ds)
def load_xy(dataset, device):
    with torch.no_grad():
        X = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device).float()
        y = torch.tensor([dataset[i][1] for i in range(len(dataset))]).to(device).long()
    return X, y

# Faz forward e pega logits e preds
def get_logits_preds(model, X):
    with torch.no_grad():
        logits = model(X)                  # [N, C] saídas brutas por classe (logits)
        preds = logits.argmax(dim=1)       # [N] classe prevista (índice do maior logit)
    return logits, preds

# Constrói matriz de confusão CxC (linhas=verdadeiro, colunas=previsto)
def confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.cpu(), y_pred.cpu()):
        cm[t, p] += 1
    return cm

# Calcula métricas por classe a partir da matriz de confusão
def per_class_metrics_from_cm(cm):
    tp = cm.diag() # verdadeiros positivos
    fp = cm.sum(0) - tp # falsos positivos
    fn = cm.sum(1) - tp # falsos negativos
    support = cm.sum(1) # numero total de exemplos reais

    eps = 1e-12
    precision = tp.float() / (tp + fp).clamp(min=1).float() # quantas precisões são corretas?
    recall    = tp.float() / (tp + fn).clamp(min=1).float()
    f1        = (2 * precision * recall) / (precision + recall + eps) #  harmônica de precision e recall
    acc_class = tp.float() / support.clamp(min=1).float()  # = recall no caso clássico

    # Macro/Micro
    macro = {
        "precision": precision.mean().item(),
        "recall":    recall.mean().item(),
        "f1":        f1.mean().item()
    }
    micro_tp = tp.sum().item()
    micro_total = cm.sum().item()
    micro = {
        "precision": micro_tp / max(1, micro_total),
        "recall":    micro_tp / max(1, micro_total),
        "f1":        micro_tp / max(1, micro_total),
        "acc":       micro_tp / max(1, micro_total),
    }

    return {
        "precision": precision,  # tensores por classe
        "recall":    recall,
        "f1":        f1,
        "acc_class": acc_class,
        "support":   support,
        "macro":     macro,
        "micro":     micro
    }

# Top confusões (erros mais frequentes verdade→previsto)
def top_confusions(cm, k=10):
    C = cm.size(0)
    pairs = []
    for i in range(C):
        for j in range(C):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j].item(), i, j))
    pairs.sort(reverse=True)
    return pairs[:min(k, len(pairs))]

############## VISUAL

# Tabela arrumada (sem pandas)
def print_metrics_table(metrics, class_names=None):
    prec, rec, f1, acc, sup = (metrics[k] for k in ["precision","recall","f1","acc_class","support"])
    C = len(prec)
    names = class_names if (class_names and len(class_names)==C) else [f"class_{i}" for i in range(C)]

    header = f"{'class':<20} {'support':>8} {'prec':>8} {'rec':>8} {'f1':>8} {'acc_cls':>8}"
    print("\nPer-class metrics:")
    print(header)
    for i in range(C):
        print(f"{names[i]:<20} {int(sup[i]):>8} {prec[i].item():>8.3f} {rec[i].item():>8.3f} {f1[i].item():>8.3f} {acc[i].item():>8.3f}")

    m, mi = metrics["macro"], metrics["micro"]
    print("\nAverages:")
    print(f"macro: prec={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    print(f"micro: prec={mi['precision']:.3f} recall={mi['recall']:.3f} f1={mi['f1']:.3f} acc={mi['acc']:.3f}")

#  Figura: heatmap da matriz de confusão (opção normalizar por linha)
def plot_confusion_matrix(cm, class_names=None, normalize=False, title="Confusion matrix"):
    cm_np = cm.numpy().astype(float)
    if normalize:
        row_sums = cm_np.sum(axis=1, keepdims=True)
        row_sums[row_sums==0] = 1.0
        cm_np = cm_np / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_np, aspect='auto')  # sem escolher paleta explicitamente
    ax.set_title(title)
    C = cm_np.shape[0]
    names = class_names if (class_names and len(class_names)==C) else [f"class_{i}" for i in range(C)]
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(C)); ax.set_yticks(np.arange(C))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)

    # Anotações nas células
    fmt = ".2f" if normalize else "d"
    for i in range(C):
        for j in range(C):
            txt = f"{cm_np[i, j]:{fmt}}"
            ax.text(j, i, txt, ha="center", va="center")

    fig.tight_layout()
    plt.show()

# D) Gráfico de barras das top confusões
def plot_top_confusions(cm, class_names=None, k=10, title="Top confusions"):
    import numpy as np
    import matplotlib.pyplot as plt

    pairs = top_confusions(cm, k=k)
    if not pairs:
        print("\n(No confusions off-diagonal found.)")
        return

    C = cm.size(0)
    names = class_names if (class_names and len(class_names)==C) else [f"class_{i}" for i in range(C)]

    labels = [f"{names[i]} → {names[j]}" for _, i, j in pairs]
    counts = [n for n, _, _ in pairs]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35*len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, counts)                 # barras horizontais
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()                  # maior no topo
    ax.set_xlabel("Count")
    ax.set_title(title)

    # anota valor na barra
    for idx, v in enumerate(counts):
        ax.text(v, idx, f" {v}", va="center")

    fig.tight_layout()
    plt.show()

def evaluate_split(model, dataset, device, class_names=None, normalize_cm=True, title=""):
    X, y = load_xy(dataset, device)
    logits, preds = get_logits_preds(model, X)
    C = logits.size(1)
    cm = confusion_matrix(y, preds, C)
    metrics = per_class_metrics_from_cm(cm)
    print_metrics_table(metrics, class_names)
    plot_confusion_matrix(cm, class_names, normalize=normalize_cm,
                          title=title or ("Confusion matrix (normalized)" if normalize_cm else "Confusion matrix"))
    # NOVO: gráfico de barras para top confusões
    plot_top_confusions(cm, class_names=class_names, k=10, title="Top confusions (true → predicted)")
    return cm, metrics

# vendo os arquivos

def top_confusion_files(dataset, preds, cm, n_files=5):
    """
    Retorna { (true_cls, pred_cls): [arquivos] } para os pares de confusão mais frequentes.
    """
    if not hasattr(dataset, "ids"):
        raise ValueError("O dataset precisa ter atributo .ids com os nomes dos arquivos.")

    ids_array = np.array(dataset.ids)
    C = cm.size(0)
    result = {}

    # pares (true != pred) ordenados por contagem decrescente
    pairs = sorted(
        [(cm[i, j].item(), i, j) for i in range(C) for j in range(C) if i != j and cm[i, j] > 0],
        reverse=True
    )

    for _, i_true, j_pred in pairs:
        arquivos = [
            ids_array[idx]
            for idx in range(len(dataset))
            if dataset.y[idx].item() == i_true and preds[idx] == j_pred
        ]
        result[(i_true, j_pred)] = arquivos[:n_files]

    return result


def top_confusion_files_from_split(model, dataset, device, cm, n_files=5):
    """Calcula preds internamente e retorna arquivos das top confusões."""
    X, y = load_xy(dataset, device)
    _, preds = get_logits_preds(model, X)
    return top_confusion_files(dataset, preds.cpu().numpy(), cm, n_files=n_files)

