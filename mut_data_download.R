# tcga mutation data download

library(TCGAbiolinks)

query_mut <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Simple Nucleotide Variation",
  data.type = "Masked Somatic Mutation",
  access = "open" 
)

GDCdownload(query_mut, method = "client")

mut_data <- GDCprepare(query = query_mut)

write.csv(mut_data, "tcga_mutations.csv")