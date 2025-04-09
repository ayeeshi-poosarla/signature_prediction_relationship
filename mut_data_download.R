library(TCGAbiolinks)

query_mut <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Simple Nucleotide Variation",
  data.type = "Masked Somatic Mutation",
  access = "open"  # needs to be open access I think
)

GDCdownload(query_mut, method = "client")

mut_data <- GDCprepare(query = query_mut)

write.csv(mut_data, "tcga_mutations.csv")