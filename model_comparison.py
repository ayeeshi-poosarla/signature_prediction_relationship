import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Import model modules
from random_forest_model import analyze_with_random_forest
from neural_network_model import analyze_with_neural_network
from linear_regression import analyze_mutational_signatures_and_immune_infiltration
from svm_model import analyze_with_svm

def create_comparison_log(filename="model_comparison_results.txt"):
    """Create a log file to save comparison results."""
    with open(filename, 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("======================\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return filename

def write_to_comparison_log(log_filename, text, print_to_console=True):
    """Write text to comparison log file and optionally print to console."""
    with open(log_filename, 'a') as f:
        f.write(text + "\n")
    if print_to_console:
        print(text)

def compare_all_models():
    """Run all four models and compare their performance."""
    comparison_log = create_comparison_log()
    
    write_to_comparison_log(comparison_log, "Starting comprehensive model comparison...")
    write_to_comparison_log(comparison_log, "This analysis will compare Logistic Regression, Random Forest, Neural Network, and SVM models")
    write_to_comparison_log(comparison_log, "=" * 80)
    
    results = {}
    
    # Run Logistic Regression
    write_to_comparison_log(comparison_log, "\n1. Running Logistic Regression model...\n")
    try:
        start_time = time.time()
        lr_model, lr_features, lr_scaler = analyze_mutational_signatures_and_immune_infiltration()
        lr_time = time.time() - start_time
        
        # Extract results from the logistic regression output file
        with open("analysis_results.txt", 'r') as f:
            lr_log = f.read()
        
        # Extract accuracy and ROC-AUC from the log file using string manipulation
        lr_accuracy = float([line for line in lr_log.split('\n') if "Model Accuracy:" in line][0].split(': ')[1])
        lr_roc_auc = float([line for line in lr_log.split('\n') if "ROC-AUC Score:" in line][0].split(': ')[1])
        lr_cv_acc = float([line for line in lr_log.split('\n') if "Cross-validation accuracy:" in line][0].split(': ')[1].split(' ±')[0])
        
        results['Logistic Regression'] = {
            'Accuracy': lr_accuracy,
            'ROC-AUC': lr_roc_auc,
            'CV Accuracy': lr_cv_acc,
            'Runtime (s)': lr_time,
            'Features': len(lr_features)
        }
        
        write_to_comparison_log(comparison_log, f"  Completed in {lr_time:.2f} seconds")
        write_to_comparison_log(comparison_log, f"  Accuracy: {lr_accuracy:.4f}")
        write_to_comparison_log(comparison_log, f"  ROC-AUC: {lr_roc_auc:.4f}")
        write_to_comparison_log(comparison_log, f"  CV Accuracy: {lr_cv_acc:.4f}")
    except Exception as e:
        write_to_comparison_log(comparison_log, f"  Error running Logistic Regression: {str(e)}")
    
    # Run Random Forest
    write_to_comparison_log(comparison_log, "\n2. Running Random Forest model...\n")
    try:
        start_time = time.time()
        rf_results = analyze_with_random_forest()
        rf_time = time.time() - start_time
        
        results['Random Forest'] = {
            'Accuracy': rf_results['accuracy'],
            'ROC-AUC': rf_results['roc_auc'],
            'CV Accuracy': rf_results['cv_accuracy'],
            'Runtime (s)': rf_time,
            'Features': len(rf_results['selected_features'])
        }
        
        write_to_comparison_log(comparison_log, f"  Completed in {rf_time:.2f} seconds")
        write_to_comparison_log(comparison_log, f"  Accuracy: {rf_results['accuracy']:.4f}")
        write_to_comparison_log(comparison_log, f"  ROC-AUC: {rf_results['roc_auc']:.4f}")
        write_to_comparison_log(comparison_log, f"  CV Accuracy: {rf_results['cv_accuracy']:.4f}")
    except Exception as e:
        write_to_comparison_log(comparison_log, f"  Error running Random Forest: {str(e)}")
    
    # Run Neural Network
    write_to_comparison_log(comparison_log, "\n3. Running Neural Network model...\n")
    try:
        start_time = time.time()
        nn_results = analyze_with_neural_network()
        nn_time = time.time() - start_time
        
        results['Neural Network'] = {
            'Accuracy': nn_results['accuracy'],
            'ROC-AUC': nn_results['roc_auc'],
            'CV Accuracy': nn_results['cv_accuracy'],
            'Runtime (s)': nn_time,
            'Features': len(nn_results['selected_features'])
        }
        
        write_to_comparison_log(comparison_log, f"  Completed in {nn_time:.2f} seconds")
        write_to_comparison_log(comparison_log, f"  Accuracy: {nn_results['accuracy']:.4f}")
        write_to_comparison_log(comparison_log, f"  ROC-AUC: {nn_results['roc_auc']:.4f}")
        write_to_comparison_log(comparison_log, f"  CV Accuracy: {nn_results['cv_accuracy']:.4f}")
    except Exception as e:
        write_to_comparison_log(comparison_log, f"  Error running Neural Network: {str(e)}")
    
    # Run SVM
    write_to_comparison_log(comparison_log, "\n4. Running SVM model...\n")
    try:
        start_time = time.time()
        svm_results = analyze_with_svm()
        svm_time = time.time() - start_time
        
        results['SVM'] = {
            'Accuracy': svm_results['accuracy'],
            'ROC-AUC': svm_results['roc_auc'],
            'CV Accuracy': svm_results['cv_accuracy'],
            'Runtime (s)': svm_time,
            'Features': len(svm_results['selected_features'])
        }
        
        write_to_comparison_log(comparison_log, f"  Completed in {svm_time:.2f} seconds")
        write_to_comparison_log(comparison_log, f"  Accuracy: {svm_results['accuracy']:.4f}")
        write_to_comparison_log(comparison_log, f"  ROC-AUC: {svm_results['roc_auc']:.4f}")
        write_to_comparison_log(comparison_log, f"  CV Accuracy: {svm_results['cv_accuracy']:.4f}")
    except Exception as e:
        write_to_comparison_log(comparison_log, f"  Error running SVM: {str(e)}")
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    write_to_comparison_log(comparison_log, "\n\nModel Comparison Summary:")
    write_to_comparison_log(comparison_log, "=" * 80)
    write_to_comparison_log(comparison_log, comparison_df.to_string())
    
    # Determine best model based on different metrics
    if comparison_df.shape[0] > 0:
        best_accuracy_model = comparison_df['Accuracy'].idxmax()
        best_roc_auc_model = comparison_df['ROC-AUC'].idxmax()
        best_cv_model = comparison_df['CV Accuracy'].idxmax()
        fastest_model = comparison_df['Runtime (s)'].idxmin()
        
        write_to_comparison_log(comparison_log, "\n\nBest Model Analysis:")
        write_to_comparison_log(comparison_log, f"Best model by Accuracy: {best_accuracy_model} ({comparison_df.loc[best_accuracy_model, 'Accuracy']:.4f})")
        write_to_comparison_log(comparison_log, f"Best model by ROC-AUC: {best_roc_auc_model} ({comparison_df.loc[best_roc_auc_model, 'ROC-AUC']:.4f})")
        write_to_comparison_log(comparison_log, f"Best model by CV Accuracy: {best_cv_model} ({comparison_df.loc[best_cv_model, 'CV Accuracy']:.4f})")
        write_to_comparison_log(comparison_log, f"Fastest model: {fastest_model} ({comparison_df.loc[fastest_model, 'Runtime (s)']:.2f} seconds)")
    
    # Generate comparison plots
    plot_comparison_metrics(results, comparison_log)
    
    write_to_comparison_log(comparison_log, "\nModel comparison completed successfully.")
    return comparison_df

def plot_comparison_metrics(results, comparison_log):
    """Generate comparison plots for the models."""
    if not results:
        write_to_comparison_log(comparison_log, "No results to plot.")
        return
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results).T
    
    # Plot accuracy metrics
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Accuracy comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x=df.index, y='Accuracy', data=df)
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)  # Assuming accuracy is between 0.5 and 1.0
    
    # Plot 2: ROC-AUC comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x=df.index, y='ROC-AUC', data=df)
    plt.title('ROC-AUC Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    # Plot 3: CV Accuracy comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x=df.index, y='CV Accuracy', data=df)
    plt.title('Cross-Validation Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    # Plot 4: Runtime comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x=df.index, y='Runtime (s)', data=df)
    plt.title('Model Runtime (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison_metrics.png')
    plt.close()
    
    write_to_comparison_log(comparison_log, "Model comparison plots saved to 'model_comparison_metrics.png'")

if __name__ == "__main__":
    print("Starting comprehensive model comparison...")
    comparison_results = compare_all_models()
    print("Comparison completed. See 'model_comparison_results.txt' for detailed results.")