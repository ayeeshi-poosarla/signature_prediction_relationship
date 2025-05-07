import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import re
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Import model modules
from random_forest_model import analyze_with_random_forest
from neural_network_model import analyze_with_neural_network
from logistic_regression import analyze_mutational_signatures_and_immune_infiltration
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

def extract_top_features(log_file, model_type):
    """Extract top features from model log files."""
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        top_features = []
        
        if model_type == "Logistic Regression":
            # Extract features from logistic regression output
            feature_section = re.search(r'Feature Importance.*?(\n.*?){10}', log_content, re.DOTALL)
            if feature_section:
                feature_lines = feature_section.group(0).split('\n')[2:12]  # Get top 10 features
                for line in feature_lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            feature = ' '.join(parts[:-1])
                            coef = float(parts[-1])
                            top_features.append((feature, coef))
        
        elif model_type == "Random Forest":
            # Extract features from random forest output
            feature_section = re.search(r'Feature Importance.*?\n(.*?\n){11}', log_content, re.DOTALL)
            if feature_section:
                feature_lines = feature_section.group(0).split('\n')[2:12]
                for line in feature_lines:
                    if line.strip() and '  ' in line:
                        parts = line.strip().split('  ')
                        parts = [p for p in parts if p.strip()]
                        if len(parts) >= 2:
                            feature = parts[0].strip()
                            importance = float(parts[-1].strip())
                            top_features.append((feature, importance))
        
        elif model_type == "Neural Network":
            # Extract selected features from neural network output
            feature_lines = re.findall(r'Selected.*?top features.*?\n(.*?\n){10}', log_content, re.DOTALL)
            if feature_lines:
                lines = feature_lines[0].split('\n')[:10]  # Get top 10 features
                for line in lines:
                    if line.strip() and ':' in line:
                        feature = line.strip().split(':')[0].strip()
                        score = float(re.search(r'Score=([\d\.]+)', line).group(1))
                        top_features.append((feature, score))
        
        elif model_type == "SVM":
            # Extract selected features from SVM output
            feature_section = re.search(r'Selected.*?top features.*?\n(.*?\n){10}', log_content, re.DOTALL)
            if feature_section:
                feature_lines = feature_section.group(0).split('\n')[1:11]
                for line in feature_lines:
                    if line.strip() and ':' in line:
                        parts = line.strip().split(':')
                        feature = parts[0].strip()
                        if 'Score=' in line:
                            score = float(re.search(r'Score=([\d\.]+)', line).group(1))
                            top_features.append((feature, score))
        
        return top_features
    
    except Exception as e:
        print(f"Error extracting features for {model_type}: {str(e)}")
        return []

def compare_all_models():
    """Run all four models and compare their performance."""
    comparison_log = create_comparison_log()
    
    write_to_comparison_log(comparison_log, "Starting comprehensive model comparison...")
    write_to_comparison_log(comparison_log, "This analysis will compare Logistic Regression, Random Forest, Neural Network, and SVM models")
    write_to_comparison_log(comparison_log, "=" * 80)
    
    results = {}
    feature_importance = {}
    
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
        
        # Extract top features
        lr_top_features = extract_top_features("analysis_results.txt", "Logistic Regression")
        feature_importance['Logistic Regression'] = lr_top_features
        
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
        
        # Extract top features
        rf_top_features = extract_top_features("rf_results.txt", "Random Forest")
        feature_importance['Random Forest'] = rf_top_features
        
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
        
        # Extract top features
        nn_top_features = extract_top_features("nn_results.txt", "Neural Network")
        feature_importance['Neural Network'] = nn_top_features
        
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
        
        # Extract top features
        svm_top_features = extract_top_features("svm_results.txt", "SVM")
        feature_importance['SVM'] = svm_top_features
        
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
    
    # Add feature importance comparison
    write_to_comparison_log(comparison_log, "\n\nTop Predictive Features by Model:")
    write_to_comparison_log(comparison_log, "=" * 80)
    
    for model_name, features in feature_importance.items():
        write_to_comparison_log(comparison_log, f"\n{model_name} - Top Features:")
        if features:
            for i, (feature, importance) in enumerate(features, 1):
                write_to_comparison_log(comparison_log, f"  {i}. {feature}: {importance:.6f}")
        else:
            write_to_comparison_log(comparison_log, "  No features extracted")
    
    # Find common important features across models
    write_to_comparison_log(comparison_log, "\n\nCommon Important Features Across Models:")
    write_to_comparison_log(comparison_log, "=" * 80)
    
    # Extract all feature names
    all_features = set()
    for model, features in feature_importance.items():
        all_features.update([feature[0] for feature in features])
    
    # Count model occurrences for each feature
    feature_counts = {feature: 0 for feature in all_features}
    for model, features in feature_importance.items():
        for feature, _ in features:
            feature_counts[feature] += 1
    
    # List features that appear in multiple models
    common_features = {f: count for f, count in feature_counts.items() if count > 1}
    if common_features:
        write_to_comparison_log(comparison_log, "Features appearing in multiple models:")
        for feature, count in sorted(common_features.items(), key=lambda x: x[1], reverse=True):
            write_to_comparison_log(comparison_log, f"  {feature}: appears in {count} models")
    else:
        write_to_comparison_log(comparison_log, "No common features found across models")
    
    # Generate comparison plots
    plot_comparison_metrics(results, comparison_log)
    
    # Generate additional visualizations
    plot_feature_importance_comparison(feature_importance, comparison_log)
    plot_mutation_signature_analysis(feature_importance, comparison_log)
    
    write_to_comparison_log(comparison_log, "\nModel comparison completed successfully.")
    return comparison_df, feature_importance

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
    bars = sns.barplot(x=df.index, y='Accuracy', data=df)
    # Add values on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(bar.get_x() + bar.get_width()/2., 
                 bar.get_height() + 0.01, 
                 f'{df.iloc[i]["Accuracy"]:.3f}', 
                 ha='center')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)  # Assuming accuracy is between 0.5 and 1.0
    
    # Plot 2: ROC-AUC comparison
    plt.subplot(2, 2, 2)
    bars = sns.barplot(x=df.index, y='ROC-AUC', data=df)
    # Add values on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(bar.get_x() + bar.get_width()/2., 
                 bar.get_height() + 0.01, 
                 f'{df.iloc[i]["ROC-AUC"]:.3f}', 
                 ha='center')
    plt.title('ROC-AUC Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    # Plot 3: CV Accuracy comparison
    plt.subplot(2, 2, 3)
    bars = sns.barplot(x=df.index, y='CV Accuracy', data=df)
    # Add values on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(bar.get_x() + bar.get_width()/2., 
                 bar.get_height() + 0.01, 
                 f'{df.iloc[i]["CV Accuracy"]:.3f}', 
                 ha='center')
    plt.title('Cross-Validation Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    # Plot 4: Runtime comparison
    plt.subplot(2, 2, 4)
    bars = sns.barplot(x=df.index, y='Runtime (s)', data=df)
    # Add values on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(bar.get_x() + bar.get_width()/2., 
                 bar.get_height() + 0.01, 
                 f'{df.iloc[i]["Runtime (s)"]:.1f}s', 
                 ha='center')
    plt.title('Model Runtime (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison_metrics.png', dpi=300)
    plt.close()
    
    write_to_comparison_log(comparison_log, "Model comparison plots saved to 'model_comparison_metrics.png'")

def plot_feature_importance_comparison(feature_importance, comparison_log):
    """Generate a heatmap comparing important features across models."""
    if not feature_importance:
        return
    
    # Get all unique features
    all_features = set()
    for model, features in feature_importance.items():
        all_features.update([feature[0] for feature in features])
    
    # Prepare data for heatmap
    heatmap_data = []
    for feature in all_features:
        row = {'Feature': feature}
        for model, features in feature_importance.items():
            feature_dict = dict(features)
            row[model] = feature_dict.get(feature, 0)
        heatmap_data.append(row)
    
    df = pd.DataFrame(heatmap_data)
    df = df.set_index('Feature')
    
    # Filter to features that appear in at least two models
    feature_counts = df.astype(bool).sum(axis=1)
    common_features = df.loc[feature_counts >= 2]
    
    if not common_features.empty:
        plt.figure(figsize=(12, len(common_features)/2 + 4))
        
        # Normalize each column to make comparison easier
        normalized_df = common_features.copy()
        for col in normalized_df.columns:
            if normalized_df[col].max() > 0:
                normalized_df[col] = normalized_df[col] / normalized_df[col].max()
        
        # Plot heatmap
        sns.heatmap(normalized_df, cmap='YlGnBu', linewidths=0.5, annot=True, fmt='.2f')
        plt.title('Common Important Features Across Models\n(Normalized Importance Scores)')
        plt.tight_layout()
        plt.savefig('common_features_heatmap.png', dpi=300)
        plt.close()
        
        write_to_comparison_log(comparison_log, "Common features heatmap saved to 'common_features_heatmap.png'")
    
    # Create a plot for top 5 features for each model
    plt.figure(figsize=(15, 12))
    
    models = list(feature_importance.keys())
    num_models = len(models)
    
    for i, model in enumerate(models, 1):
        if feature_importance[model]:
            # Get top 5 features for this model
            features = feature_importance[model][:5]
            feature_names = [f[0] for f in features]
            importance = [f[1] for f in features]
            
            plt.subplot(num_models, 1, i)
            bars = plt.barh(range(len(feature_names)), importance, align='center')
            plt.yticks(range(len(feature_names)), feature_names)
            plt.title(f'Top 5 Features - {model}')
            plt.tight_layout()
            
            # Add values to the end of bars
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{width:.4f}', ha='left', va='center')
    
    plt.tight_layout(pad=3)
    plt.savefig('top_features_by_model.png', dpi=300)
    plt.close()
    
    write_to_comparison_log(comparison_log, "Top features by model saved to 'top_features_by_model.png'")

def plot_mutation_signature_analysis(feature_importance, comparison_log):
    """Analyze and visualize mutation signatures across models."""
    if not feature_importance:
        return
    
    # Extract mutation signature features (SNV and Type)
    mutation_features = {}
    for model, features in feature_importance.items():
        mutation_features[model] = []
        for feature, importance in features:
            # Check if feature is a mutation signature
            if 'Type_' in feature or 'SNV_' in feature or 'Norm_Type_' in feature or 'Norm_SNV_' in feature:
                mutation_features[model].append((feature, importance))
    
    # Create visualization if mutation features were found
    if any(mutation_features.values()):
        plt.figure(figsize=(14, 10))
        
        # Prepare data
        model_names = []
        feature_lists = []
        importance_lists = []
        
        for model, features in mutation_features.items():
            if features:
                for feature, importance in features:
                    model_names.append(model)
                    feature_lists.append(feature)
                    importance_lists.append(importance)
        
        # Create DataFrame
        mutation_df = pd.DataFrame({
            'Model': model_names,
            'Feature': feature_lists,
            'Importance': importance_lists
        })
        
        # Plot mutation signatures across models
        plt.subplot(1, 2, 1)
        sns.barplot(x='Feature', y='Importance', hue='Model', data=mutation_df)
        plt.title('Mutation Signatures Across Models')
        plt.xticks(rotation=90)
        
        # Create a count of mutation signature types
        plt.subplot(1, 2, 2)
        feature_counts = mutation_df['Feature'].value_counts().head(10)
        feature_counts.plot(kind='bar')
        plt.title('Most Common Mutation Signatures')
        plt.xlabel('Mutation Signature')
        plt.ylabel('Count (Across Models)')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig('mutation_signature_analysis.png', dpi=300)
        plt.close()
        
        write_to_comparison_log(comparison_log, "Mutation signature analysis saved to 'mutation_signature_analysis.png'")
        
        # Create a detailed analysis of mutation signatures
        write_to_comparison_log(comparison_log, "\n\nMutation Signature Analysis:")
        write_to_comparison_log(comparison_log, "=" * 80)
        
        # Analyze common mutation signatures
        common_mutations = mutation_df['Feature'].value_counts()
        common_mutations = common_mutations[common_mutations > 1]
        
        if not common_mutations.empty:
            write_to_comparison_log(comparison_log, "\nMost predictive mutation signatures (appearing in multiple models):")
            for feature, count in common_mutations.items():
                # Get models that include this feature
                models_with_feature = mutation_df[mutation_df['Feature'] == feature]['Model'].unique()
                avg_importance = mutation_df[mutation_df['Feature'] == feature]['Importance'].mean()
                
                write_to_comparison_log(comparison_log, f"  {feature}: appears in {count} models ({', '.join(models_with_feature)}), avg importance: {avg_importance:.4f}")
        
        # Analyze mutation types vs SNV types
        type_mutations = mutation_df[mutation_df['Feature'].str.contains('Type_')]
        snv_mutations = mutation_df[mutation_df['Feature'].str.contains('SNV_')]
        
        write_to_comparison_log(comparison_log, f"\nMutation types (count: {len(type_mutations)}), SNV types (count: {len(snv_mutations)})")
        
        if not type_mutations.empty:
            avg_importance_types = type_mutations['Importance'].mean()
            write_to_comparison_log(comparison_log, f"Average importance of mutation types: {avg_importance_types:.4f}")
        
        if not snv_mutations.empty:
            avg_importance_snv = snv_mutations['Importance'].mean()
            write_to_comparison_log(comparison_log, f"Average importance of SNV types: {avg_importance_snv:.4f}")

if __name__ == "__main__":
    print("Starting comprehensive model comparison...")
    comparison_results, feature_importance = compare_all_models()
    print("Comparison completed. See 'model_comparison_results.txt' for detailed results.")