Best K: 2

Number of participants per cluster (K-means, best K):
  Cluster 0: 182
  Cluster 1: 77

============================================================
VARIABLE IMPORTANCE ANALYSIS
============================================================

1. CLUSTER CENTER ANALYSIS:
----------------------------------------
Variables ranked by standard deviation of cluster centers:
  SRS_social_communication_tscore          0.858
  SRS_social_cognition_tscore              0.843
  SRS_restrictive_repetitive_tscore        0.826
  ASEBA_aggressive_behavior_tscore         0.585
  ASEBA_attention_problems_tscore          0.558
  ASEBA_externalizing_problems_tscore      0.413
  ASEBA_anxious_depressed_tscore           0.380
  ASEBA_internalizing_problems_tscore      0.289
  SCQ_score                                0.146

2. ANOVA F-SCORES:
----------------------------------------
Variables ranked by ANOVA F-scores:
  SRS_social_communication_tscore          F=411.783, p=0.0000 ***
  SRS_social_cognition_tscore              F=376.491, p=0.0000 ***
  SRS_restrictive_repetitive_tscore        F=339.921, p=0.0000 ***
  ASEBA_aggressive_behavior_tscore         F=103.055, p=0.0000 ***
  ASEBA_attention_problems_tscore          F=90.218, p=0.0000 ***
  ASEBA_externalizing_problems_tscore      F=42.720, p=0.0000 ***
  ASEBA_anxious_depressed_tscore           F=35.232, p=0.0000 ***
  ASEBA_internalizing_problems_tscore      F=19.225, p=0.0000 ***
  SCQ_score                                F=4.670, p=0.0316 *

3. RANDOM FOREST FEATURE IMPORTANCE:
----------------------------------------
Variables ranked by Random Forest importance:
  SRS_social_communication_tscore          0.315
  SRS_social_cognition_tscore              0.298
  SRS_restrictive_repetitive_tscore        0.153
  ASEBA_attention_problems_tscore          0.066
  ASEBA_aggressive_behavior_tscore         0.056
  ASEBA_externalizing_problems_tscore      0.050
  ASEBA_internalizing_problems_tscore      0.031
  ASEBA_anxious_depressed_tscore           0.023
  SCQ_score                                0.008

4. CLUSTER MEANS BY VARIABLE:
----------------------------------------
Mean values by cluster (original scale):
         SRS_social_cognition_tscore  ...  SCQ_score
Cluster                               ...           
0                              50.43  ...      17.58
1                              72.65  ...      19.09

[2 rows x 9 columns]

Variables ranked by range across clusters:
  SRS_restrictive_repetitive_tscore        23.15
  SRS_social_communication_tscore          22.26
  SRS_social_cognition_tscore              22.22
  ASEBA_internalizing_problems_tscore      10.09
  ASEBA_attention_problems_tscore          8.51
  ASEBA_aggressive_behavior_tscore         7.84
  ASEBA_externalizing_problems_tscore      5.24
  ASEBA_anxious_depressed_tscore           4.69
  SCQ_score                                1.51

5. CREATING VISUALIZATIONS...

============================================================
SUMMARY: TOP PREDICTIVE VARIABLES
============================================================
Variables ranked by average across all methods:
   2. SRS_social_communication_tscore     (avg rank: 0.0)
   1. SRS_social_cognition_tscore         (avg rank: 1.0)
   3. SRS_restrictive_repetitive_tscore   (avg rank: 2.0)
   6. ASEBA_aggressive_behavior_tscore    (avg rank: 3.3)
   7. ASEBA_attention_problems_tscore     (avg rank: 3.7)
   5. ASEBA_externalizing_problems_tscore (avg rank: 5.0)
   8. ASEBA_anxious_depressed_tscore      (avg rank: 6.3)
   4. ASEBA_internalizing_problems_tscore (avg rank: 6.7)
   9. SCQ_score                           (avg rank: 8.0)