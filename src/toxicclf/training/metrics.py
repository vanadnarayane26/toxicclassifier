import torch
from sklearn.metrics import f1_score

def compute_f1_score(y_true, y_pred, threshold=0.5):
    probs = torch.sigmoid(y_pred)
    y_pred_labels = (probs > threshold).int().cpu().numpy()
    y_true = y_true.cpu().numpy()
    micro_f1 = f1_score(y_true, y_pred_labels, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
    return micro_f1, macro_f1