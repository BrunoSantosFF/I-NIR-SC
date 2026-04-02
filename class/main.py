import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split


from model import CombinedMLP
from data_loader import load_and_prepare_data, MultiModalDataset
from plot import plot_loss_accuracy, plot_confusion_matrix


BASE_PATH = "caminho relativo"
VAE_FEATURES_PATH = f"{BASE_PATH}/I-NIR-SC/inputs/vetor_latente.txt"
OUTPUT_PATH = f"{BASE_PATH}/classificacao_combinada"

MODEL_CONFIGS = {
    'densenet-121': {
        'data': "/home/bruno/test/tcc/I-NIR-SC/inputs/features_modelo1.txt",
        'display_name': 'Densenet121'
    },
}

USE_IMPROVED_ATTENTION = False
COMBINATION_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9]
N_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_EPOCHS_CV = 500
NUM_EPOCHS_FINAL = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 5
N_TEST_RUNS = 5


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for cnn_x, vae_x, y in val_loader:
            cnn_x, vae_x, y = cnn_x.to(device), vae_x.to(device), y.to(device)
            outputs, _ = model(cnn_x, vae_x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    return 100 * correct / total, val_loss / len(val_loader)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for cnn_x, vae_x, y in test_loader:
            cnn_x, vae_x, y = cnn_x.to(device), vae_x.to(device), y.to(device)
            out, _ = model(cnn_x, vae_x)
            probs = torch.nn.functional.softmax(out, dim=1)
            _, preds = out.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    
    class_names = np.unique(all_labels)
    n_classes = len(class_names)
    if n_classes > 2:
        y_onehot = label_binarize(all_labels, classes=class_names)
        auc_macro = roc_auc_score(y_onehot, all_probs, multi_class='ovr', average='macro')
    else:
        auc_macro = roc_auc_score(all_labels, all_probs[:, 1])
    
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'auc_macro': auc_macro,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'class_names': [str(c) for c in class_names]
    }


def train_model_with_cv(model, train_data, val_data, device, num_epochs, class_weights=None):
    train_ds = MultiModalDataset(train_data['cnn'], train_data['vae'], train_data['labels'])
    val_ds = MultiModalDataset(val_data['cnn'], val_data['vae'], val_data['labels'])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model.to(device)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=1e-5)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for cnn_x, vae_x, y in train_loader:
            cnn_x, vae_x, y = cnn_x.to(device), vae_x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(cnn_x, vae_x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
        
        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        val_acc, val_loss = validate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f} | Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} | Val Acc={val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    metrics = evaluate_model(model, val_loader, device)
    metrics['history'] = history
    metrics['dims_info'] = model.get_dims_info()
    
    return model, metrics


def run_cv_for_cf(cf, data, model_config, device):
    print(f"\n{'='*60}")
    print(f"CROSS VALIDATION FOR CF={cf}")
    print(f"{'='*60}")
    
    cnn_features = data['cnn_train']
    vae_features = data['vae_train']
    labels = data['labels_train']
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_metrics = []
    fold_histories = []
    all_confusion_matrices = []
    
    class_counts = np.bincount(labels)
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(cnn_features, labels), 1):
        print(f"\nFOLD {fold}/{N_FOLDS}")
        
        train_data = {
            'cnn': cnn_features[train_idx],
            'vae': vae_features[train_idx],
            'labels': labels[train_idx]
        }
        
        val_data = {
            'cnn': cnn_features[val_idx],
            'vae': vae_features[val_idx],
            'labels': labels[val_idx]
        }
        
        model = CombinedMLP(
            data['cnn_dim'],
            data['vae_dim'],
            len(np.unique(labels)),
            combination_factor=cf,
            use_attention=USE_IMPROVED_ATTENTION
        )
        
        trained_model, metrics = train_model_with_cv(
            model, train_data, val_data, device,
            num_epochs=NUM_EPOCHS_CV, class_weights=weights
        )
        
        metrics['fold'] = fold
        fold_metrics.append(metrics)
        fold_histories.append(metrics['history'])
        all_confusion_matrices.append(metrics['confusion_matrix'])
        
        print(f"Fold {fold} - Accuracy: {metrics['accuracy']:.4f}, Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    
    aggregated_metrics = {}
    metric_keys = ['accuracy', 'balanced_accuracy', 'auc_macro', 'precision', 'recall', 'f1_score']
    
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        aggregated_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    aggregated_cm = np.sum(all_confusion_matrices, axis=0)
    
    return {
        'cf': cf,
        'fold_metrics': fold_metrics,
        'aggregated_metrics': aggregated_metrics,
        'aggregated_confusion_matrix': aggregated_cm,
        'fold_histories': fold_histories
    }


def run_multiple_tests(data, best_cf, device, num_classes, n_runs=N_TEST_RUNS):
    print(f"\n{'='*60}")
    print(f"TRAINING {n_runs} INDEPENDENT MODELS")
    print(f"{'='*60}")
    
    test_ds = MultiModalDataset(data['cnn_test'], data['vae_test'], data['labels_test'])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_metrics = []
    all_confusion_matrices = []
    
    for run in range(n_runs):
        print(f"\n{'='*40}")
        print(f"RUN {run+1}/{n_runs}")
        print(f"{'='*40}")
        
        run_random_state = RANDOM_STATE + run * 10
        
        model = CombinedMLP(
            data['cnn_dim'],
            data['vae_dim'],
            num_classes,
            combination_factor=best_cf,
            use_attention=USE_IMPROVED_ATTENTION
        )
        model.to(device)
        
        X_train, X_val, vae_train, vae_val, y_train, y_val = train_test_split(
            data['cnn_train'], 
            data['vae_train'],
            data['labels_train'],
            test_size=0.15,
            stratify=data['labels_train'], 
            random_state=run_random_state
        )
        
        class_counts = np.bincount(y_train)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        class_weights = torch.FloatTensor(weights).to(device)
        
        train_dataset = MultiModalDataset(X_train, vae_train, y_train)
        val_dataset = MultiModalDataset(X_val, vae_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        best_epoch = 0
        
        for epoch in range(NUM_EPOCHS_FINAL):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for cnn_x, vae_x, y in train_loader:
                cnn_x, vae_x, y = cnn_x.to(device), vae_x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs, _ = model(cnn_x, vae_x)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
            
            train_acc = 100 * correct / total
            avg_train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for cnn_x, vae_x, y in val_loader:
                    cnn_x, vae_x, y = cnn_x.to(device), vae_x.to(device), y.to(device)
                    outputs, _ = model(cnn_x, vae_x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(y).sum().item()
                    total += y.size(0)
            
            val_acc = 100 * correct / total
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
                best_epoch = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        
        metrics = evaluate_model(model, test_loader, device)
        metrics['run'] = run + 1
        metrics['best_epoch'] = best_epoch
        metrics['best_val_loss'] = best_val_loss
        all_metrics.append(metrics)
        all_confusion_matrices.append(metrics['confusion_matrix'])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    avg_metrics = {}
    metric_keys = ['accuracy', 'balanced_accuracy', 'auc_macro', 'precision', 'recall', 'f1_score']
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    aggregated_cm = np.sum(all_confusion_matrices, axis=0)
    
    return {
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics,
        'aggregated_cm': aggregated_cm
    }


def run_model_experiment(model_key, model_config):
    print("\n" + "="*70)
    print(f"STARTING EXPERIMENT: {model_config['display_name']}")
    print("="*70)
    
    model_output_path = f"{OUTPUT_PATH}/{model_key}"
    os.makedirs(model_output_path, exist_ok=True)
    print(f"Results will be saved in: {model_output_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data = load_and_prepare_data(model_config, VAE_FEATURES_PATH, TEST_SIZE, RANDOM_STATE)
    num_classes = len(np.unique(data['labels_train']))
    print(f"Number of classes: {num_classes}")
    
    print("\nEvaluating different Combination Factors with CV...")
    cv_results = []
    
    for cf in COMBINATION_FACTORS:
        cv_result = run_cv_for_cf(cf, data, model_config, device)
        cv_results.append(cv_result)
        
        cf_path = f"{model_output_path}/cf_{cf}"
        os.makedirs(cf_path, exist_ok=True)
        
        fold_metrics_df = pd.DataFrame([
            {**m, 'fold': i+1}
            for i, m in enumerate(cv_result['fold_metrics'])
        ])
        fold_metrics_df.to_csv(f"{cf_path}/fold_metrics.csv", index=False)
        
        agg_metrics_df = pd.DataFrame([{
            'cf': cf,
            **{f"{k}_mean": v['mean'] for k, v in cv_result['aggregated_metrics'].items()},
            **{f"{k}_std": v['std'] for k, v in cv_result['aggregated_metrics'].items()}
        }])
        agg_metrics_df.to_csv(f"{cf_path}/aggregated_metrics.csv", index=False)
        
        agg_metrics = {'confusion_matrix': cv_result['aggregated_confusion_matrix']}
        plot_confusion_matrix(agg_metrics, model_config['display_name'], cf, cf_path)
        
        for fold, history in enumerate(cv_result['fold_histories'], 1):
            plot_loss_accuracy(history, model_config['display_name'], cf, cf_path, fold)
    
    best_cf = None
    best_bal_acc = -1
    
    for cv_result in cv_results:
        mean_bal_acc = cv_result['aggregated_metrics']['balanced_accuracy']['mean']
        if mean_bal_acc > best_bal_acc:
            best_bal_acc = mean_bal_acc
            best_cf = cv_result['cf']
    
    print(f"\nBEST CF FOUND: {best_cf}")
    print(f"Balanced Accuracy Mean: {best_bal_acc:.4f}")
    
    cf_comparison = []
    for cv_result in cv_results:
        cf_comparison.append({
            'cf': cv_result['cf'],
            'accuracy_mean': cv_result['aggregated_metrics']['accuracy']['mean'],
            'accuracy_std': cv_result['aggregated_metrics']['accuracy']['std'],
            'balanced_accuracy_mean': cv_result['aggregated_metrics']['balanced_accuracy']['mean'],
            'balanced_accuracy_std': cv_result['aggregated_metrics']['balanced_accuracy']['std'],
            'auc_macro_mean': cv_result['aggregated_metrics']['auc_macro']['mean'],
            'auc_macro_std': cv_result['aggregated_metrics']['auc_macro']['std'],
            'f1_score_mean': cv_result['aggregated_metrics']['f1_score']['mean'],
            'f1_score_std': cv_result['aggregated_metrics']['f1_score']['std']
        })
    
    cf_comparison_df = pd.DataFrame(cf_comparison)
    cf_comparison_df.to_csv(f"{model_output_path}/cf_comparison_cv.csv", index=False)
    
    print(f"\nRunning {N_TEST_RUNS} test executions for average...")
    multi_test_results = run_multiple_tests(data, best_cf, device, num_classes, N_TEST_RUNS)
    
    test_path = f"{model_output_path}/test_results_cf_{best_cf}_{N_TEST_RUNS}runs"
    os.makedirs(test_path, exist_ok=True)
    
    all_test_metrics = []
    for i, metrics in enumerate(multi_test_results['all_metrics']):
        test_metrics_df = pd.DataFrame([{
            'run': i+1,
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'auc_macro': metrics['auc_macro'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }])
        all_test_metrics.append(test_metrics_df)
    
    all_test_metrics_df = pd.concat(all_test_metrics, ignore_index=True)
    all_test_metrics_df.to_csv(f"{test_path}/all_test_runs.csv", index=False)
    
    avg_metrics_dict = {
        'metric': [],
        'mean': [],
        'std': []
    }
    for key, value in multi_test_results['avg_metrics'].items():
        avg_metrics_dict['metric'].append(key)
        avg_metrics_dict['mean'].append(value['mean'])
        avg_metrics_dict['std'].append(value['std'])
    
    avg_metrics_df = pd.DataFrame(avg_metrics_dict)
    avg_metrics_df.to_csv(f"{test_path}/average_metrics_{N_TEST_RUNS}runs.csv", index=False)
    
    agg_metrics = {'confusion_matrix': multi_test_results['aggregated_cm']}
    plot_confusion_matrix(agg_metrics, model_config['display_name'], best_cf, test_path, fold=f'test_{N_TEST_RUNS}runs')
    
    print(f"\nTEST RESULTS AVERAGE ({N_TEST_RUNS} runs):")
    for key, value in multi_test_results['avg_metrics'].items():
        print(f"- {key}: {value['mean']:.4f} +/- {value['std']:.4f}")
    
    summary = {
        'model_name': model_config['display_name'],
        'best_combination_factor': best_cf,
        'n_folds': N_FOLDS,
        'n_test_runs': N_TEST_RUNS,
        'cv_results': {
            cf: {
                'balanced_accuracy_mean': r['aggregated_metrics']['balanced_accuracy']['mean'],
                'balanced_accuracy_std': r['aggregated_metrics']['balanced_accuracy']['std'],
                'accuracy_mean': r['aggregated_metrics']['accuracy']['mean'],
                'accuracy_std': r['aggregated_metrics']['accuracy']['std']
            } for cf, r in zip(COMBINATION_FACTORS, cv_results)
        },
        'test_results_average': {
            key: {'mean': value['mean'], 'std': value['std']}
            for key, value in multi_test_results['avg_metrics'].items()
        },
        'all_test_runs': [
            {'run': i+1, 'accuracy': m['accuracy'], 'balanced_accuracy': m['balanced_accuracy']}
            for i, m in enumerate(multi_test_results['all_metrics'])
        ],
        'aggregated_confusion_matrix': multi_test_results['aggregated_cm'].tolist(),
        'cnn_features': data['cnn_dim'],
        'vae_features': data['vae_dim'],
        'train_samples': len(data['cnn_train']),
        'test_samples': len(data['cnn_test']),
        'num_classes': num_classes,
        'use_attention': USE_IMPROVED_ATTENTION,
        'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{model_output_path}/experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    with open(f"{model_output_path}/EXPERIMENT_SUMMARY.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"EXPERIMENT SUMMARY - {model_config['display_name']}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best CF found: {best_cf}\n")
        f.write(f"Number of folds: {N_FOLDS}\n")
        f.write(f"Number of test runs: {N_TEST_RUNS}\n\n")
        
        f.write("CROSS VALIDATION RESULTS (mean +/- std):\n")
        f.write("-"*50 + "\n")
        for cf, result in zip(COMBINATION_FACTORS, cv_results):
            bal_acc = result['aggregated_metrics']['balanced_accuracy']
            acc = result['aggregated_metrics']['accuracy']
            f.write(f"\nCF={cf}:\n")
            f.write(f"  Balanced Acc: {bal_acc['mean']:.4f} +/- {bal_acc['std']:.4f}\n")
            f.write(f"  Accuracy: {acc['mean']:.4f} +/- {acc['std']:.4f}\n")
        
        f.write("\n\nTEST SET RESULTS (average over {} runs):\n".format(N_TEST_RUNS))
        f.write("-"*50 + "\n")
        for key, value in multi_test_results['avg_metrics'].items():
            f.write(f"{key}: {value['mean']:.4f} +/- {value['std']:.4f}\n")
        
        f.write("\nConfusion Matrix (aggregated over {} runs):\n".format(N_TEST_RUNS))
        cm = multi_test_results['aggregated_cm']
        f.write("               PREDICTED\n")
        f.write("             (0)     (1)\n")
        f.write(f"TRUE   (0)   {cm[0,0]:5d}   {cm[0,1]:5d}\n")
        f.write(f"       (1)   {cm[1,0]:5d}   {cm[1,1]:5d}\n")
    
    print(f"\nResults saved in: {model_output_path}")
    print("="*70)
    
    return summary


def main():
    print("="*80)
    print("MULTIMODAL CLASSIFICATION SYSTEM WITH CROSS VALIDATION")
    print("="*80)
    print(f"Base directory: {BASE_PATH}")
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"Available models: {len(MODEL_CONFIGS)}")
    print(f"Combination Factors: {COMBINATION_FACTORS}")
    print(f"Cross Validation: {N_FOLDS} folds")
    print(f"Test runs: {N_TEST_RUNS}")
    print(f"Using Attention: {USE_IMPROVED_ATTENTION}")
    print("="*80)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print("\nAVAILABLE MODELS:")
    for i, (key, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"{i}. {config['display_name']} ({key})")
    
    print(f"{len(MODEL_CONFIGS) + 1}. ALL models")
    
    try:
        choice = input("\nChoose model (number or 'all'): ").strip().lower()
        
        if choice == 'all' or choice == str(len(MODEL_CONFIGS) + 1):
            all_summaries = []
            for model_key, model_config in MODEL_CONFIGS.items():
                try:
                    summary = run_model_experiment(model_key, model_config)
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"\nError in model {model_config['display_name']}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if all_summaries:
                summary_data = []
                for s in all_summaries:
                    row = {
                        'model_name': s['model_name'],
                        'best_cf': s['best_combination_factor'],
                        'test_balanced_acc_mean': s['test_results_average']['balanced_accuracy']['mean'],
                        'test_balanced_acc_std': s['test_results_average']['balanced_accuracy']['std'],
                        'test_accuracy_mean': s['test_results_average']['accuracy']['mean'],
                        'test_accuracy_std': s['test_results_average']['accuracy']['std'],
                        'train_samples': s['train_samples'],
                        'test_samples': s['test_samples']
                    }
                    summary_data.append(row)
                
                all_summaries_df = pd.DataFrame(summary_data)
                all_summaries_df.to_csv(f"{OUTPUT_PATH}/all_models_summary.csv", index=False)
                print(f"\nAll models summary saved in: {OUTPUT_PATH}/all_models_summary.csv")
        
        elif choice.isdigit() and 1 <= int(choice) <= len(MODEL_CONFIGS):
            model_idx = int(choice) - 1
            model_key = list(MODEL_CONFIGS.keys())[model_idx]
            model_config = MODEL_CONFIGS[model_key]
            run_model_experiment(model_key, model_config)
        else:
            print("Invalid choice. Please run again.")
    
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()