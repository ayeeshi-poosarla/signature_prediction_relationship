import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Create a log file for saving results
def create_log_file(filename="svm_results.txt"):
    """Create a log file to save analysis results."""
    with open(filename, 'w') as f:
        f.write("SVM Classification Results\n")
        f.write("========================\n\n")
    return filename

# Function to write to log file
def write_to_log(log_filename, text, print_to_console=True):
    """Write text to log file and optionally print to console."""
    with open(log_filename, 'a') as f:
        f.write(text + "\n")
    if print_to_console:
        print(text)

# Load the dataset
def load_data(filepath, log_filename):
    """Load and prepare the dataset from CSV file."""
    write_to_log(log_filename, "Loading and preparing data...")
    df = pd.read_csv(filepath)
    write_to_log(log_filename, f"Dataset loaded with {df.shape[0]} samples and {df.shape[1]} features.")
    return df

# Preprocess the data
def preprocess_data(df, log_filename):
    """Preprocess the dataset by handling missing values and splitting features."""
    # Handle missing values if any
    if df.isna().sum().any():
        write_to_log(log_filename, f"Found {df.isna().sum().sum()} missing values. Filling with appropriate values...")
        df = df.fillna(0)  # Simple imputation strategy
    
    # Extract patient IDs for reference
    patient_ids = df['Patient_ID']
    
    # Extract mutation features (regular and normalized)
    mutation_types = df.filter(regex='^Type_').columns
    snv_types = df.filter(regex='^SNV_').columns
    norm_mutation_types = df.filter(regex='^Norm_Type_').columns
    norm_snv_types = df.filter(regex='^Norm_SNV_').columns
    
    # Extract gene expression features - these are genes with ENSG IDs
    gene_expr_cols = df.filter(regex='^ENSG').columns
    
    # Create feature groups
    mutation_features = df[list(mutation_types) + list(snv_types)]
    normalized_mutation_features = df[list(norm_mutation_types) + list(norm_snv_types)]
    gene_expression_features = df[gene_expr_cols]
    total_mutations = df[['Total_Mutations']]
    
    write_to_log(log_filename, f"Extracted {len(mutation_types) + len(snv_types)} mutation features")
    write_to_log(log_filename, f"Extracted {len(norm_mutation_types) + len(norm_snv_types)} normalized mutation features")
    write_to_log(log_filename, f"Extracted {len(gene_expr_cols)} gene expression features")
    
    return patient_ids, total_mutations, mutation_features, normalized_mutation_features, gene_expression_features

# Define target variable based on immune infiltration levels
def define_immune_infiltration_target(df, gene_expr_features, log_filename, immune_marker_genes=None):
    """
    Create a target variable based on immune cell infiltration signatures.
    Using gene expression profiles of known immune marker genes.
    """
    if immune_marker_genes is None:
        write_to_log(log_filename, "No specific immune marker genes provided. Using PCA to identify patterns...")
        
        # Use PCA to identify main patterns in gene expression
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(gene_expr_features)
        
        # Convert PCA score to binary target (high/low immune infiltration)
        # Using median as threshold for simplicity
        median_score = np.median(principal_component)
        target = (principal_component > median_score).astype(int).flatten()
        
        write_to_log(log_filename, f"Created target variable using PCA. Class distribution: {np.bincount(target)}")
    else:
        # If specific immune marker genes are provided
        immune_signature = gene_expr_features[immune_marker_genes].mean(axis=1)
        target = (immune_signature > immune_signature.median()).astype(int)
        write_to_log(log_filename, f"Created target variable using specified immune marker genes. Class distribution: {np.bincount(target)}")
    
    return target

# Feature selection function
def select_features(X, y, log_filename, k=15):
    """Select top k features using ANOVA F-value."""
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    write_to_log(log_filename, f"Selected {k} top features:")
    for i, feature in enumerate(selected_features):
        f_score = selector.scores_[selector.get_support()][i]
        p_value = selector.pvalues_[selector.get_support()][i]
        write_to_log(log_filename, f"  {feature}: F-score={f_score:.4f}, p-value={p_value:.4e}")
    
    return X_new, selected_features

# Build, train and evaluate the SVM model
def build_svm_model(X_train, X_test, y_train, y_test, feature_names, log_filename):
    """Build and evaluate an SVM model."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search for hyperparameter tuning
    write_to_log(log_filename, "Performing grid search for hyperparameter tuning...")
    param_grid = {
        'C': [0.1, 1, 10],        # Reduced from 4 values
        'gamma': ['scale', 'auto'], # Removed numerical values
        'kernel': ['rbf']         # Focus on one kernel type first
    }
    model = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    write_to_log(log_filename, f"Best hyperparameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    write_to_log(log_filename, f"\nModel Accuracy: {accuracy:.4f}")
    
    write_to_log(log_filename, "\nClassification Report:")
    write_to_log(log_filename, classification_report(y_test, y_pred))
    
    # For linear kernels, we can interpret coefficients
    if best_model.kernel == 'linear':
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': best_model.coef_[0]
        })
        feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
        
        write_to_log(log_filename, "\nFeature Coefficients (top 10 and bottom 10):")
        importance_table = pd.concat([feature_importance.head(10), feature_importance.tail(10)]).to_string()
        write_to_log(log_filename, importance_table)
    
    return best_model, y_pred, y_pred_proba, scaler

# Plot ROC curve
def plot_roc_curve(y_test, y_pred_proba, log_filename):
    """Plot ROC curve for model evaluation."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
    plt.legend(loc="lower right")
    plt.savefig('svm_roc_curve.png')
    plt.close()
    
    write_to_log(log_filename, f"ROC-AUC Score: {roc_auc:.4f}")
    write_to_log(log_filename, "ROC curve has been saved to 'svm_roc_curve.png'")
    
    return roc_auc

# Visualize confusion matrix
def plot_confusion_matrix(y_test, y_pred, log_filename):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - SVM')
    plt.savefig('svm_confusion_matrix.png')
    plt.close()
    
    write_to_log(log_filename, "Confusion matrix has been saved to 'svm_confusion_matrix.png'")

# Main function to run SVM analysis
def analyze_with_svm():
    """Run analysis with SVM model."""
    # Create log file
    log_filename = create_log_file()
    
    # Data loading and preprocessing
    filepath = 'ml_dataset.csv'
    df = load_data(filepath, log_filename)
    
    patient_ids, total_mutations, mutation_features, normalized_mutation_features, gene_expression_features = preprocess_data(df, log_filename)
    
    # Define target variable based on immune infiltration
    target = define_immune_infiltration_target(df, gene_expression_features, log_filename)
    
    write_to_log(log_filename, "\n--- Using normalized mutation features for SVM model ---")
    
    # Combine relevant features (using normalized mutation features and total mutations)
    X = pd.concat([total_mutations, normalized_mutation_features], axis=1)
    y = target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    write_to_log(log_filename, f"Training set: {X_train.shape[0]} samples")
    write_to_log(log_filename, f"Testing set: {X_test.shape[0]} samples")
    
    # Select the most informative features
    X_train_selected, selected_features = select_features(X_train, y_train, log_filename, k=20)
    X_test_selected = X_test[selected_features]
    
    # Build and evaluate the model
    model, y_pred, y_pred_proba, scaler = build_svm_model(
        X_train_selected, X_test_selected, y_train, y_test, selected_features, log_filename
    )
    
    # Visualization
    roc_auc = plot_roc_curve(y_test, y_pred_proba, log_filename)
    plot_confusion_matrix(y_test, y_pred, log_filename)
    
    # Perform cross-validation to check for overfitting
    write_to_log(log_filename, "\nPerforming cross-validation...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5
    svm_model = SVC(
        C=model.C,
        gamma=model.gamma,
        kernel=model.kernel,
        probability=True,
        random_state=42
    )
    cv_scores = cross_val_score(
        svm_model,
        X[selected_features], y, cv=cv, scoring='accuracy'
    )
    
    write_to_log(log_filename, f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return {
        'model': model,
        'selected_features': selected_features,
        'scaler': scaler,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc,
        'cv_accuracy': cv_scores.mean()
    }

# Entry point of the script
if __name__ == "__main__":
    log_filename = create_log_file()
    write_to_log(log_filename, "Starting analysis with SVM model...")
    results = analyze_with_svm()
    write_to_log(log_filename, "\nSVM analysis completed successfully.")