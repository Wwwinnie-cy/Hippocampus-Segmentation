import os
import numpy as np
import nibabel as nib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_nifti(file_path):
    return nib.load(file_path).get_fdata()

def calculate_metrics(pred_dir, true_dir):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])
    true_files = sorted([f for f in os.listdir(true_dir) if f.endswith('.nii.gz')])

    all_labels = []
    all_preds = []

    for pred_file, true_file in zip(pred_files, true_files):
        pred_path = os.path.join(pred_dir, pred_file)
        true_path = os.path.join(true_dir, true_file)

        pred = load_nifti(pred_path).astype(np.int32)
        true = load_nifti(true_path).astype(np.int32)

        all_labels.extend(true.flatten())
        all_preds.extend(pred.flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    pred_dir = './saved_nii_results'
    true_dir = './labelsTs'
    calculate_metrics(pred_dir, true_dir)
