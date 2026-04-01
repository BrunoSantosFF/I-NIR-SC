import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_loss_accuracy(history, model_name, cf, output_path, fold=None):
    fold_str = f"_fold_{fold}" if fold is not None else ""
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = f'{model_name} CF={cf}'
    title += f' - Fold {fold}' if fold is not None else ''
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/loss_accuracy{fold_str}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(metrics, model_name, cf, output_path, fold=None):
    cm = metrics['confusion_matrix']
    class_names = ['Saudavel (0)', 'Cancer (1)']
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('TRUE', fontsize=12, fontweight='bold')
    plt.xlabel('PREDICTED', fontsize=12, fontweight='bold')
    
    if fold is not None:
        plt.title(f'{model_name} CF={cf} - Fold {fold}\nConfusion Matrix', fontsize=14, fontweight='bold')
    else:
        plt.title(f'{model_name} CF={cf}\nConfusion Matrix (Aggregated)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if fold is not None:
        plt.savefig(f"{output_path}/confusion_matrix_fold_{fold}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{output_path}/confusion_matrix_aggregated.png", dpi=300, bbox_inches='tight')
    
    plt.close()