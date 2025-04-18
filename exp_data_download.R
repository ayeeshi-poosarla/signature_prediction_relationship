# download tcga expression data

library(TCGAbiolinks)

query_exp <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)
GDCdownload(query_exp)
exp_data <- GDCprepare(query = query_exp)

exp_df <- as.data.frame(SummarizedExperiment::assay(exp_data))

write.csv(exp_df, "tcga_expression.csv")