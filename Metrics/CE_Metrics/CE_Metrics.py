from pprint import pprint
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


def compute_mlc(gt, pred, label_set):
    res_mlc = {}
    avg_aucroc = 0
    for i, label in enumerate(label_set):
        res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
        avg_aucroc += res_mlc['AUCROC_' + label]
    res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)

    res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc


def main():
    """
    This function is designed to handle the extraction of labels from the MIMIC-CXR dataset.
    It reads the labeled result data and the ground truth labels from CSV files, processes them,
    and calculates metrics to evaluate the performance based on these labels.
    """
    res_path = "Data/MIMIC-CXR/res_labeled.csv data"   #  mimic_cxr/res_labeled.csv data
    gts_path = "Data/MIMIC-CXR/gts_labeled.csv data"   #  mimic_cxr/gts_labeled.csv data
    res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
    res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

    label_set = res_data.columns[1:].tolist()
    res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
    res_data[res_data == -1] = 0
    gts_data[gts_data == -1] = 0

    metrics = compute_mlc(gts_data, res_data, label_set)
    pprint(metrics)


if __name__ == '__main__':
    main()