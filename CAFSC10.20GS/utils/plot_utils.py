import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_util(cm, class_names, epoch, output_dir="confusion_matrices_cafsc"):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues,
                xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"cm_epoch_{epoch + 1}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[Info] Confusion matrix saved to {save_path}")
