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

# code failed below here, new fix present on mut_data_download.R file
query_mut <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Simple Nucleotide Variation",
  data.type = "Masked Somatic Mutation"
)
GDCdownload(query_mut)
mut_data <- GDCprepare(query = query_mut)

write.csv(mut_data, "tcga_mutations.csv")