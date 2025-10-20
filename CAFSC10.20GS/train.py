import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from model.cafsc_net import CAFSC_Net
from data.augmentation import AntiNoiseAugmentation
from utils.train_utils import LabelSmoothingLoss, train_one_epoch, validate_and_get_cm_data
from utils.plot_utils import plot_confusion_matrix_util

def parse_args():
    parser = argparse.ArgumentParser(description='Train CAFSC Network (Public Review Version)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained_resnet', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'valid')

    train_transform = AntiNoiseAugmentation(image_size=224)
    val_transform = train_transform.final_transform

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = CAFSC_Net(num_classes=len(class_names), pretrained_resnet=args.pretrained_resnet).to(args.device)
    criterion = LabelSmoothingLoss(classes=len(class_names), smoothing=0.1).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc, all_labels, all_preds = validate_and_get_cm_data(model, val_loader, criterion, args.device)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_public_cafsc.pth')
            print('[Info] Saved new best checkpoint.')

        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix_util(cm, class_names, epoch)

if __name__ == '__main__':
    main()
