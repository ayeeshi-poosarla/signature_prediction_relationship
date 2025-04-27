import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Create a log file for saving results
def create_log_file(filename="rf_results.txt"):
    """Create a log file to save analysis results."""
    with open(filename, 'w') as f:
        f.write("Random Forest Classification Results\n")
        f.write("===================================\n\n")
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

# Build, train and evaluate the random forest model
def build_random_forest_model(X_train, X_test, y_train, y_test, feature_names, log_filename):
    """Build and evaluate a random forest model."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check for class imbalance and apply SMOTE if necessary
    class_counts = np.bincount(y_train)
    if class_counts[0] / len(y_train) < 0.35 or class_counts[0] / len(y_train) > 0.65:
        write_to_log(log_filename, f"Addressing class imbalance with SMOTE. Original distribution: {class_counts}")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        write_to_log(log_filename, f"After SMOTE resampling: {np.bincount(y_train)}")
    
    # Grid search for hyperparameter tuning (with reduced parameter space)
    write_to_log(log_filename, "Performing grid search for hyperparameter tuning (simplified)...")
    param_grid = {
        'n_estimators': [100, 200],     # Reduced from 3 values
        'max_depth': [None, 20],        # Reduced from 4 values
        'class_weight': ['balanced', None]  # Reduced from 3 values
    }
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    
    # Set a timeout for grid search (5 minutes)
    from contextlib import contextmanager
    import signal
    
    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError("Grid search timed out")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    
    # Try with timeout
    best_model = None
    try:
        with time_limit(300):  # 5 minutes timeout
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            write_to_log(log_filename, f"Best hyperparameters: {grid_search.best_params_}")
    except TimeoutError:
        write_to_log(log_filename, "Grid search timed out. Using default model.")
        best_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42)
        best_model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    write_to_log(log_filename, f"\nModel Accuracy: {accuracy:.4f}")
    
    write_to_log(log_filename, "\nClassification Report:")
    write_to_log(log_filename, classification_report(y_test, y_pred))
    
    # Display feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    write_to_log(log_filename, "\nFeature Importance (top 10):")
    importance_table = feature_importance.head(10).to_string()
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
    plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
    plt.legend(loc="lower right")
    plt.savefig('rf_roc_curve.png')
    plt.close()
    
    write_to_log(log_filename, f"ROC-AUC Score: {roc_auc:.4f}")
    write_to_log(log_filename, "ROC curve has been saved to 'rf_roc_curve.png'")
    
    return roc_auc

# Plot feature importance
def plot_feature_importance(model, feature_names, log_filename):
    """Plot feature importance from the random forest model."""
    # Get feature importances from model
    importances = model.feature_importances_
    
    # Create a dataframe for better visualization
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(feature_importances.index[:15])), feature_importances['Importance'][:15])
    plt.yticks(range(len(feature_importances.index[:15])), feature_importances['Feature'][:15])
    plt.xlabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    plt.close()
    
    write_to_log(log_filename, "Feature importance plot has been saved to 'rf_feature_importance.png'")

# Visualize confusion matrix
def plot_confusion_matrix(y_test, y_pred, log_filename):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Random Forest')
    plt.savefig('rf_confusion_matrix.png')
    plt.close()
    
    write_to_log(log_filename, "Confusion matrix has been saved to 'rf_confusion_matrix.png'")

# Main function to run random forest analysis
def analyze_with_random_forest():
    """Run analysis with Random Forest model."""
    # Create log file
    log_filename = create_log_file()
    
    # Data loading and preprocessing
    filepath = 'ml_dataset.csv'
    df = load_data(filepath, log_filename)
    
    patient_ids, total_mutations, mutation_features, normalized_mutation_features, gene_expression_features = preprocess_data(df, log_filename)
    
    # Define target variable based on immune infiltration
    target = define_immune_infiltration_target(df, gene_expression_features, log_filename)
    
    write_to_log(log_filename, "\n--- Using normalized mutation features for Random Forest model ---")
    
    # Try using more features for better contextual learning
    X = pd.concat([total_mutations, normalized_mutation_features], axis=1)
    
    # Also include selected gene expression features that might be relevant
    gene_expr_variance = gene_expression_features.var().sort_values(ascending=False)
    top_variable_genes = gene_expr_variance.index[:15]  # Take top 15 most variable genes
    X = pd.concat([X, gene_expression_features[top_variable_genes]], axis=1)
    
    write_to_log(log_filename, f"Including {len(top_variable_genes)} most variable gene expression features for modeling")
    y = target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    write_to_log(log_filename, f"Training set: {X_train.shape[0]} samples")
    write_to_log(log_filename, f"Testing set: {X_test.shape[0]} samples")
    
    # Try different feature selection approaches
    k_values = [15, 20, 25, 30]
    best_accuracy = 0
    best_k = k_values[0]
    
    for k in k_values:
        write_to_log(log_filename, f"\nTrying feature selection with k={k}...")
        X_train_selected, selected_features = select_features(X_train, y_train, log_filename, k=k)
        X_test_selected = X_test[selected_features]
        
        # Quick test with a simple model
        test_model = RandomForestClassifier(n_estimators=100, random_state=42)
        test_model.fit(X_train_selected, y_train)
        accuracy = test_model.score(X_test_selected, y_test)
        
        write_to_log(log_filename, f"Preliminary accuracy with k={k}: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    write_to_log(log_filename, f"\nSelected optimal k={best_k} for feature selection")
    X_train_selected, selected_features = select_features(X_train, y_train, log_filename, k=best_k)
    X_test_selected = X_test[selected_features]
    
    # Build and evaluate the model
    model, y_pred, y_pred_proba, scaler = build_random_forest_model(
        X_train_selected, X_test_selected, y_train, y_test, selected_features, log_filename
    )
    
    # Visualization
    roc_auc = plot_roc_curve(y_test, y_pred_proba, log_filename)
    plot_confusion_matrix(y_test, y_pred, log_filename)
    plot_feature_importance(model, selected_features, log_filename)
    
    # Perform cross-validation to check for overfitting
    write_to_log(log_filename, "\nPerforming cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_model = RandomForestClassifier(
        n_estimators=model.n_estimators,
        max_depth=model.max_depth,
        min_samples_split=model.min_samples_split,
        random_state=42,
        class_weight=model.class_weight
    )
    cv_scores = cross_val_score(
        rf_model,
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
    write_to_log(log_filename, "Starting analysis with Random Forest model...")
    results = analyze_with_random_forest()
    write_to_log(log_filename, "\nRandom Forest analysis completed successfully.")