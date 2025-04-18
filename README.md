# signature_prediction_relationship

Steps to produce data

1. run bioconductor_download.R
2. run data_download.R --> tcga_expression.csv
3. run mut_data_download.R

Steps to produce ML dataset
1. run environment.yml to download all dependencies
2. run data_processing.py --> ml_dataset.csv