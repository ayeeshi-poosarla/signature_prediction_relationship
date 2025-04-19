import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
def load_data(filepath):
    """Load and prepare the dataset from CSV file."""
    print("Loading and preparing data...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with {df.shape[0]} samples and {df.shape[1]} features.")
    return df

# Preprocess the data
def preprocess_data(df):
    """Preprocess the dataset by handling missing values and splitting features."""
    # Handle missing values if any
    if df.isna().sum().any():
        print(f"Found {df.isna().sum().sum()} missing values. Filling with appropriate values...")
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
    
    print(f"Extracted {len(mutation_types) + len(snv_types)} mutation features")
    print(f"Extracted {len(norm_mutation_types) + len(norm_snv_types)} normalized mutation features")
    print(f"Extracted {len(gene_expr_cols)} gene expression features")
    
    return patient_ids, total_mutations, mutation_features, normalized_mutation_features, gene_expression_features

# Define target variable based on immune infiltration levels
def define_immune_infiltration_target(df, gene_expr_features, immune_marker_genes=None):
    """
    Create a target variable based on immune cell infiltration signatures.
    Using gene expression profiles of known immune marker genes.
    
    If specific immune marker genes are not provided, will use PCA on gene expression
    to identify patterns and create a binary classification target.
    """
    if immune_marker_genes is None:
        print("No specific immune marker genes provided. Using PCA to identify patterns...")
        
        # Use PCA to identify main patterns in gene expression
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(gene_expr_features)
        
        # Convert PCA score to binary target (high/low immune infiltration)
        # Using median as threshold for simplicity
        median_score = np.median(principal_component)
        target = (principal_component > median_score).astype(int).flatten()
        
        print(f"Created target variable using PCA. Class distribution: {np.bincount(target)}")
    else:
        # If specific immune marker genes are provided
        immune_signature = gene_expr_features[immune_marker_genes].mean(axis=1)
        target = (immune_signature > immune_signature.median()).astype(int)
        print(f"Created target variable using specified immune marker genes. Class distribution: {np.bincount(target)}")
    
    return target

# Feature selection function
def select_features(X, y, k=15):
    """Select top k features using ANOVA F-value."""
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    print(f"Selected {k} top features:")
    for i, feature in enumerate(selected_features):
        f_score = selector.scores_[selector.get_support()][i]
        p_value = selector.pvalues_[selector.get_support()][i]
        print(f"  {feature}: F-score={f_score:.4f}, p-value={p_value:.4e}")
    
    return X_new, selected_features

# Build, train and evaluate the logistic regression model
def build_logistic_regression_model(X_train, X_test, y_train, y_test, feature_names, C_values=None):
    """Build and evaluate a logistic regression model."""
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search for hyperparameter tuning
    print("Performing grid search for hyperparameter tuning...")
    param_grid = {'C': C_values}
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameter C: {grid_search.best_params_['C']}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Display feature importance
    coef = best_model.coef_[0]
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
    feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
    
    print("\nFeature Importance (top 10 positive and negative):")
    print(pd.concat([feature_importance.head(10), feature_importance.tail(10)]))
    
    return best_model, y_pred, y_pred_proba, scaler

# Plot ROC curve
def plot_roc_curve(y_test, y_pred_proba):
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
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("ROC curve has been saved to 'roc_curve.png'")

# Plot feature importance
def plot_feature_importance(model, feature_names):
    """Plot feature importance from the logistic regression model."""
    coef = model.coef_[0]
    
    # Get indices of top and bottom features by coefficient magnitude
    sorted_idx = np.argsort(coef)
    top_n = 15  # Show top 15 most important features in each direction
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(2*top_n), 
             np.concatenate([coef[sorted_idx[:top_n]], coef[sorted_idx[-top_n:]]]), 
             align='center')
    plt.yticks(range(2*top_n), 
               np.array(feature_names)[np.concatenate([sorted_idx[:top_n], sorted_idx[-top_n:]])])
    plt.xlabel('Coefficient Magnitude')
    plt.title('Top Features Impact on Immune Cell Infiltration')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Feature importance plot has been saved to 'feature_importance.png'")

# Visualize confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix has been saved to 'confusion_matrix.png'")

# Main analysis function
def analyze_mutational_signatures_and_immune_infiltration():
    """Main function to run the analysis."""
    # Data loading and preprocessing
    filepath = 'ml_dataset.csv'
    df = load_data(filepath)
    
    patient_ids, total_mutations, mutation_features, normalized_mutation_features, gene_expression_features = preprocess_data(df)
    
    # Define target variable based on immune infiltration
    # For this example, we'll use immune-related genes or use PCA if none specified
    # Note: In a real analysis, you would want to use known immune marker genes
    target = define_immune_infiltration_target(df, gene_expression_features)
    
    print("\n--- Using normalized mutation features for the model ---")
    
    # Combine relevant features (using normalized mutation features and total mutations)
    X = pd.concat([total_mutations, normalized_mutation_features], axis=1)
    y = target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Select the most informative features
    X_train_selected, selected_features = select_features(X_train, y_train, k=20)
    X_test_selected = X_test[selected_features]
    
    # Build and evaluate the model
    model, y_pred, y_pred_proba, scaler = build_logistic_regression_model(X_train_selected, X_test_selected, y_train, y_test, selected_features)
    
    # Visualization
    plot_roc_curve(y_test, y_pred_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, selected_features)
    
    # Perform cross-validation to check for overfitting
    print("\nPerforming cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LogisticRegression(C=model.C, solver='liblinear', max_iter=1000),
        X[selected_features], y, cv=cv, scoring='accuracy'
    )
    
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model, selected_features, scaler

# Entry point of the script
if __name__ == "__main__":
    print("Starting analysis of how mutational signatures shape immune cell infiltration...")
    model, selected_features, scaler = analyze_mutational_signatures_and_immune_infiltration()
    print("\nAnalysis completed successfully.")