import pandas as pd

# 1. Load raw counts (genes × aliquots)
expr = pd.read_csv("tcga_expression.csv", index_col=0)

# 2. Trim each column to the patient ID (first 12 chars)
expr.columns = expr.columns.str[:12]

# 3. If you have multiple aliquots per patient, average them:
expr_by_patient = expr.groupby(axis=1, level=0).mean()

# expr_by_patient is now genes × patients