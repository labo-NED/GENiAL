########################## Install packages ##########################
install.packages('readxl')
install.packages("sjmisc")
install.packages('jmv') # ANOVA
install.packages("haven")

########################## Activate packages #########################
library(readxl)
library(sjmisc)
library(jmv)
library(haven)

########################## Import dataset ############################
original_dataset <- read.csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/proband_clusters_kmeans_complete_cases.csv")
og_dataset_copy <- original_dataset # Make a copy for preprocessing

########################### Preprocessing ############################

#### Plausibilité des scores ###
# Check frequencies
sink(file="RAW_freq_output.txt")
lapply(og_dataset_copy, sjmisc::frq)
sink()
sink(NULL) # Disable active sink

################### Convert to numeric/factor ########################

# Check classes (numeric, factor, or heaven_labelled)
lapply(og_dataset_copy, class) #No heaven labelled, can proceed.




