import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import time

import config
from model import CAF_SI_Net, CAF_Ablation_2, CAF_Ablation_3, BaselineNet, BaselineNetWithAttention, \
    get_backbone_channels
from data import AntiNoiseAndSpatialAugmentation, StandardAugmentation
from utils import (
    LabelSmoothingLoss,
    train_one_epoch,
    validate_and_get_cm_data,
    plot_confusion_matrix_util
)


def setup_device():
    device_str = config.DEVICE
    if "cuda" in device_str and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        print(f"Warning: CUDA not available or {device_str} not found, switching to CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def initialize_optimizer(model):
    if config.OPTIMIZER_TYPE.upper() == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=config.INITIAL_LR, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER_TYPE.upper() == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.INITIAL_LR, weight_decay=config.WEIGHT_DECAY,
                              momentum=config.MOMENTUM)
    else:
        raise ValueError("Please configure a correct OPTIMIZER_TYPE ('Adam' or 'SGD') in config.py")
    return optimizer


def get_data_loaders(train_transform_cls):
    data_path = config.DATA_ROOT
    if "!!!" in data_path or not os.path.exists(data_path):
        print("\n" + "=" * 80)
        print("ERROR: Please first modify DATA_ROOT in config.py to your dataset path!")
        print(f"Current path: {data_path}")
        print("=" * 80 + "\n")
        exit()

    train_data_path = os.path.join(data_path, "train")
    valid_data_path = os.path.join(data_path, "valid")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.RAFDB_MEAN, std=config.RAFDB_STD)
    ])

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform_cls(image_size=224))
    val_dataset = datasets.ImageFolder(root=valid_data_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


def run_experiment(model_cls, train_aug_cls, exp_id, device):
    print(f"\n======== Starting Experiment {exp_id} ========")

    train_loader, val_loader = get_data_loaders(train_aug_cls)

    model = model_cls().to(device)
    optimizer = initialize_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    criterion = LabelSmoothingLoss(classes=config.NUM_CLASSES, smoothing=config.LABEL_SMOOTHING).to(device)

    best_val_acc = 0.0

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_cls.__name__}, Train Aug: {train_aug_cls.__name__}, Params: {total_params / 1e6:.2f}M")

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, all_labels_val, all_preds_val = validate_and_get_cm_data(model, val_loader, criterion,
                                                                                    device)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    print(f"======== Exp {exp_id} Finished: Best Test Acc: {best_val_acc:.2f}% ========")
    return total_params, best_val_acc


def main_ablation():
    device = setup_device()

    print("\n--- Running Table 2 Ablation Experiments ---")

    run_experiment(CAF_SI_Net, StandardAugmentation, "1", device)
    run_experiment(CAF_Ablation_2, AntiNoiseAndSpatialAugmentation, "2", device)
    run_experiment(CAF_Ablation_3, AntiNoiseAndSpatialAugmentation, "3", device)
    run_experiment(CAF_SI_Net, AntiNoiseAndSpatialAugmentation, "4", device)

    print("\n--- Running Table 3 Attention Replacement Experiments ---")

    run_experiment(BaselineNet, AntiNoiseAndSpatialAugmentation, "Baseline", device)
    run_experiment(lambda: BaselineNetWithAttention('SENet'), AntiNoiseAndSpatialAugmentation, "Baseline + SENet",
                   device)
    run_experiment(lambda: BaselineNetWithAttention('CBAM'), AntiNoiseAndSpatialAugmentation, "Baseline + CBAM", device)
    run_experiment(lambda: BaselineNetWithAttention('ECAM'), AntiNoiseAndSpatialAugmentation, "Baseline + ECAM", device)
    run_experiment(CAF_SI_Net, AntiNoiseAndSpatialAugmentation, "CAFSC", device)

    print("\nAll Ablation Experiments are configured. Please run the script to get results.")
    print("WARNING: Due to non-optimal settings in config.py (e.g., Hidden Factor=1.0, SGD, ImageNet Mean/Std),")
    print("the results will NOT match the optimal values in your paper tables.")


if __name__ == '__main__':
    main_ablation()