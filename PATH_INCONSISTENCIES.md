# Path Inconsistencies Report

## Summary
This document identifies path inconsistencies between the README, actual directory structure, and scripts.

## Major Issues Found

### 1. **CLEAN Directory Mismatch**
- **Actual structure**: `/DATA/` at repository root
- **Scripts reference**: `/CLEAN/DATA/` (doesn't exist in structure)
- **README references**: `/DATA/` (matches actual structure)

**Affected scripts:**
- All scripts in `SCRIPTS/` directory use paths like:
  - `/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/`
  - But the actual structure is: `/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/`

**Recommendation**: Either:
  - Create a `CLEAN/` directory and move `DATA/` inside it, OR
  - Update all scripts to remove `/CLEAN/` from paths

### 2. **Case Sensitivity Issues**

#### REDCAP_REPORTS vs Redcap_reports
- **README says**: `/DATA/REDCAP_REPORTS/`
- **Scripts use**: `/DATA/Redcap_reports/` (mixed case)
- **Actual structure**: `/DATA/REDCAP_REPORTS/` (all caps)

#### OUTPUTS vs Outputs
- **README says**: `/DATA/OUTPUTS/`
- **Scripts use**: `/DATA/Outputs/` (capital O)
- **Actual structure**: `/DATA/OUTPUTS/` (all caps)

**Affected scripts:**
- `preprocess_demog_beh_iq_gen.py`: Uses `Redcap_reports` and `Outputs`
- All other scripts: Use `Outputs` (capital O)

**Recommendation**: 
- On macOS (case-insensitive by default), this may work but is inconsistent
- Standardize to match actual directory structure (all caps: `REDCAP_REPORTS`, `OUTPUTS`)

### 3. **Absolute vs Relative Paths**

- **README uses**: Relative paths from repo root (e.g., `/DATA/OUTPUTS/`)
- **Scripts use**: Absolute paths with user's home directory
  - Example: `/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/`

**Recommendation**: 
- Consider using relative paths or environment variables for portability
- Or document that scripts require path updates for different users

## Detailed Path Mapping

### STEP 1 - Preprocessing
| Component | README | Scripts | Actual Structure |
|-----------|--------|---------|------------------|
| REDCap input | `/DATA/REDCAP_REPORTS/` | `/CLEAN/DATA/Redcap_reports/` | `/DATA/REDCAP_REPORTS/` |
| CNV output | `/DATA/Genetic_cnv_scores/` | Not in scripts | `/DATA/Genetic_cnv_scores/` |
| Preprocessed output | `/DATA/OUTPUTS/Preprocessed/` | `/CLEAN/DATA/Outputs/Preprocessed/` | `/DATA/OUTPUTS/Preprocessed/` |

### STEP 2 - Clustering
| Component | README | Scripts | Actual Structure |
|-----------|--------|---------|------------------|
| Input | `/DATA/OUTPUTS/Preprocessed/` | `/CLEAN/DATA/Outputs/Preprocessed/` | `/DATA/OUTPUTS/Preprocessed/` |
| Output | `/DATA/OUTPUTS/Clustered/` | `/CLEAN/DATA/Outputs/Clustered/` | `/DATA/OUTPUTS/Clustered/` |

### STEP 3 - EEG Features
| Component | README | Scripts | Actual Structure |
|-----------|--------|---------|------------------|
| EEG input | Not specified | `/CLEAN/DATA/EEG` | `/DATA/EEG/` |
| Feature output | `/DATA/OUTPUTS/eeg_features/` | `/CLEAN/DATA/Outputs/eeg_features/` | `/DATA/OUTPUTS/eeg_features/` |

### STEP 4 - Statistics
| Component | README | Scripts | Actual Structure |
|-----------|--------|---------|------------------|
| Input | `/DATA/OUTPUTS/Final/` | `/CLEAN/DATA/Outputs/merged_clustered_*.csv` | `/DATA/OUTPUTS/` (files at root) |
| Output | `/DATA/OUTPUTS/Stats/` | `/CLEAN/DATA/Outputs/Stats/` | `/DATA/OUTPUTS/Stats/` |

## Files That Need Path Updates

### Python Scripts
1. `SCRIPTS/1-initial-cleanup/preprocess_demog_beh_iq_gen.py`
   - Line 780: `ROOT_DIR = "/Users/.../GENiAL/CLEAN"`
   - Should be: `ROOT_DIR = "/Users/.../GENiAL"` or use relative path

2. `SCRIPTS/2-clustering/SOM_behavioral_clustering.py`
   - Line 15: `ROOT_DIR = "/Users/.../GENiAL/CLEAN/DATA"`
   - Should remove `/CLEAN`

3. `SCRIPTS/2-clustering/gfmm_behavioral_clustering.py`
   - Line 12: Similar issue

4. `SCRIPTS/3-EEG/compute_features.py`
   - Lines 37, 39: Uses `/CLEAN/DATA/`

5. `SCRIPTS/3-EEG/aggregate_features_by_roi.py`
   - Line 21: Uses `/CLEAN/DATA/`

6. `SCRIPTS/3-EEG/aggregate_features_global.py`
   - Line 21: Uses `/CLEAN/DATA/`

7. `SCRIPTS/3-EEG/merge_eeg_features_to_db.py`
   - Lines 13-18: Uses `/CLEAN/DATA/`

8. `SCRIPTS/3-EEG/merge_5s_features_to_db.py`
   - Lines 14-22: Uses `/CLEAN/DATA/`

### R Scripts
1. `SCRIPTS/4-statistical-analysis/genial_stats.r`
   - Line 38: Uses `/CLEAN/DATA/Outputs/`
   - Multiple other lines with same issue

2. `SCRIPTS/4-statistical-analysis/diagnosis_pie_charts.r`
   - Line 20: Uses `/CLEAN/DATA/Outputs/`

3. `SCRIPTS/4-statistical-analysis/generate_behavioral_scores_plot.r`
   - Line 14: Uses `/CLEAN/DATA/Outputs/`

4. `SCRIPTS/4-statistical-analysis/stats.r`
   - Line 16: Uses `/CLEAN/DATA/Outputs/`

## Recommendations

1. **Immediate Fix**: Update README to reflect actual script paths OR update all scripts to match README
2. **Standardization**: Choose one path convention and apply consistently
3. **Portability**: Consider using relative paths or environment variables
4. **Case Sensitivity**: Standardize directory names (prefer all caps to match actual structure)

## Action Items

- [ ] Decide: Keep `/CLEAN/` or remove it from all scripts
- [ ] Standardize case: `OUTPUTS` vs `Outputs`, `REDCAP_REPORTS` vs `Redcap_reports`
- [ ] Update all scripts to use consistent paths
- [ ] Update README to match final path structure
- [ ] Test that all scripts can find their input/output files
