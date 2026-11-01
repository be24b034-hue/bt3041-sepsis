"""Step 1-2 style preprocessing and cleaning of the three project CSVs.

Inputs  (Downloads): Genes.csv (genes x 802 samples), Genes(479).csv, Patients.csv
Outputs (cleaned/) : samples_clean.csv, genes_x_samples.csv, genes_x_sepsis479.csv,
                     genes_x_sepsis479_expressed.csv, cleaning_log.csv, qc_samples.csv
Output names and column names match the instruction manual, so Steps 3-11 run on them unchanged.
"""
import os
import numpy as np
import pandas as pd

DL = r"C:\Users\bharath\Downloads"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned")
os.makedirs(OUT, exist_ok=True)
log = []                                        # (step, decision, detail)

def note(step, decision, detail):
    log.append({"step": step, "decision": decision, "detail": detail})
    print(f"[{step}] {decision}: {detail}")

genes = pd.read_csv(os.path.join(DL, "Genes.csv"), index_col=0)
g479 = pd.read_csv(os.path.join(DL, "Genes(479).csv"), index_col=0)
pat = pd.read_csv(os.path.join(DL, "Patients.csv"), index_col=0)

# ---------------------------------------------------------------- gene table
# '---' is Affymetrix's placeholder for probe sets with no gene symbol; it is not a gene
n0 = len(genes)
genes = genes.drop(index=[g for g in genes.index if g == "---"])
note("genes", "dropped '---' row", f"{n0} -> {len(genes)} genes (probe sets without a gene symbol)")

assert genes.isna().sum().sum() == 0 and not genes.index.duplicated().any()
note("genes", "no missing values / duplicate genes", f"range {genes.values.min():.2f}-{genes.values.max():.2f} (already log2 RMA)")

same = np.allclose(genes[g479.columns].values, g479.drop(index="---").values)
note("genes", "Genes(479).csv is a column subset of Genes.csv", f"identical values: {same}; regenerated from the cleaned full table")

# ---------------------------------------------------------------- patient table
pat = pat.rename(columns={"sample_title": "title"})
pat["group"] = pat["group"].replace({"Sepsis_with_outcome": "Sepsis"})
note("patients", "renamed group 'Sepsis_with_outcome' -> 'Sepsis', 'sample_title' -> 'title'", "matches the manual's code")

pat = pat.drop(columns=["source"])
note("patients", "dropped 'source'", "constant ('whole blood' for all 802)")

pat["gender"] = pat["gender"].str.lower().str.strip()
pat["male"] = (pat["gender"] == "male").astype(int)

# outcome as integers, only defined for sepsis
pat["mortality_event_28days"] = pat["mortality_event_28days"].astype("Int64")
pat["time_to_event_28days"] = pat["time_to_event_28days"].astype("Int64")

# consistent spelling of the paired ICU-infection labels
pat["icu_acquired_infection_paired"] = pat["icu_acquired_infection_paired"].replace(
    {"Admis_ICUAI": "admission_ICUAI", "ADMIS_NO_ICUAI": "admission_no_ICUAI", "Event_ICUAI": "event_ICUAI"})
note("patients", "standardised icu_acquired_infection_paired spelling", "Admis_/ADMIS_ mixed case")

# ordinal platelet score from the letter prefix (A worst ... D normal)
pat["thrombocytopenia_score"] = pat["thrombocytopenia"].map(
    {"A_very_low": 0, "B_medium_low": 1, "C_low": 2, "D_normal": 3}).astype("Int64")

pat["diabetes"] = pat["diabetes_mellitus"].map({"DM": 1, "No_DM": 0}).astype("Int64")

# one infection-source column for sepsis patients
src = pd.Series("other/unknown", index=pat.index)
src[pat.pneumonia_diagnoses == "cap"] = "CAP"
src[pat.pneumonia_diagnoses == "hap"] = "HAP"
src[pat.abdominal_sepsis_and_controls == "abdo_s"] = "abdominal"
both = pat.pneumonia_diagnoses.isin(["cap", "hap"]) & (pat.abdominal_sepsis_and_controls == "abdo_s")
src[both] = "pneumonia+abdominal"
pat["infection_source"] = src.where(pat.group == "Sepsis")
note("patients", "derived infection_source for sepsis",
     f"{pat.infection_source.value_counts().to_dict()}; both labels: {pat.index[both & (pat.group == 'Sepsis')].tolist()}")

# batch information hidden in the sample title
pat["run_date"] = pd.to_datetime(pat["title"].str.extract(r"\[(\d\d_\d\d_\d{4})")[0], format="%d_%m_%Y")
pat["array_well"] = pat["title"].str.extract(r"\[\d\d_\d\d_\d{4}_([A-H]\d\d)")[0]
note("patients", "extracted run_date and array_well from title", f"{pat.run_date.nunique()} run dates (batch variable)")

pat["hospital"] = pat["endotype_cohort"].map({"discovery": "Amsterdam", "validation": "Utrecht"})

# ---------------------------------------------------------------- outcome sanity flags
sep = pat.group == "Sepsis"
died = pat.mortality_event_28days == 1
t = pat.time_to_event_28days
pat["flag_censored_early"] = (sep & (pat.mortality_event_28days == 0) & (t < 28)).fillna(False)
pat["flag_death_day0"] = (sep & died & (t == 0)).fillna(False)
note("outcome", "kept, flagged 2 survivors followed < 28 days",
     f"{pat.index[pat.flag_censored_early].tolist()} (censored at day "
     f"{t[pat.flag_censored_early].tolist()}); their 'survived' label is uncertain -> sensitivity analysis")
note("outcome", "kept 11 deaths at day 0", "death on the admission day; valid for Kaplan-Meier/Cox")
note("outcome", "kept 2 deaths at day 28", "death on day 28 is still within 28 days")
note("outcome", "outcome missing only outside sepsis", f"sepsis missing outcome = {pat.loc[sep, 'mortality_event_28days'].isna().sum()}")

young = pat.index[pat.age < 18].tolist()
note("patients", "kept age 17 patient", f"{young}; plausible adult-ICU admission, ages 17-93")

dm_missing = pat.loc[sep, "diabetes"].isna().sum()
note("patients", "diabetes left missing (no imputation)",
     f"{dm_missing}/479 sepsis patients ({100*dm_missing/479:.1f}%) -> complete-case in models that use it")

# ---------------------------------------------------------------- sample QC on expression
X = genes[pat.index]
corr_med = X.corrwith(X.median(axis=1))
z = (corr_med - corr_med.mean()) / corr_med.std()
pat["qc_corr_to_median"] = corr_med.round(4)
pat["flag_expr_outlier"] = z < -4
note("QC", "kept all arrays; flagged z<-4 outliers",
     f"{int(pat.flag_expr_outlier.sum())} flagged (corr to median sample {corr_med.min():.3f}-{corr_med.max():.3f}); "
     f"groups: {pat.loc[pat.flag_expr_outlier, 'group'].value_counts().to_dict()}")

# sex check: XIST is female-only, Y-chromosome genes male-only
ygenes = [g for g in ["RPS4Y1", "DDX3Y", "KDM5D", "EIF1AY", "UTY", "USP9Y"] if g in X.index]
sex_score = X.loc[ygenes].mean() - X.loc["XIST"]
# cut-offs sit in the empty gap of each bimodal distribution (XIST 3.0-4.0, RPS4Y1 4.5-6.0)
y_on = X.loc["RPS4Y1"] > 5.0
x_on = X.loc["XIST"] > 3.5
pat["sex_score"] = sex_score.round(3)
pat["expr_sex"] = np.select([y_on & ~x_on, x_on & ~y_on, y_on & x_on], ["male", "female", "mixed"], "unclear")
pat["flag_sex_mismatch"] = pat.expr_sex != pat.gender
note("QC", "sex check (RPS4Y1 vs XIST)",
     f"{int(pat.flag_sex_mismatch.sum())} samples where expression sex != recorded sex "
     f"({pat.loc[pat.flag_sex_mismatch, 'expr_sex'].value_counts().to_dict()}): "
     f"{pat.index[pat.flag_sex_mismatch].tolist()}; groups {pat.loc[pat.flag_sex_mismatch, 'group'].value_counts().to_dict()}. "
     f"Kept, flagged: possible sample/label swaps -> sensitivity analysis")

# near-duplicate arrays (same patient sampled twice, or a re-hybridised array).
# Centre genes within each group first, otherwise all healthy controls look alike.
top = X.var(axis=1).sort_values(ascending=False).index[:2000]
Xc = X.loc[top].copy()
for _, ids in pat.groupby("group").groups.items():
    Xc[ids] = Xc[ids].sub(Xc[ids].mean(axis=1), axis=0)
C = np.corrcoef(Xc.T.values)
np.fill_diagonal(C, np.nan)
iu = np.triu_indices_from(C, 1)
pairs = pd.DataFrame({"a": X.columns[iu[0]], "b": X.columns[iu[1]], "r": C[iu]}).sort_values("r", ascending=False)
pairs = pairs.reset_index(drop=True)
close = pairs[pairs.r > 0.9]
for c in ["a", "b"]:
    for col in ["group", "age", "gender"]:
        pairs[f"{c}_{col}"] = pat.loc[pairs[c], col].values
print("\nmost similar sample pairs (group-centred, top-2000 genes):")
print(pairs.head(5).round(3).to_string())
pat["max_corr_other_sample"] = np.nanmax(C, axis=1).round(3)
note("QC", "no duplicate arrays", f"pairs with r > 0.9: {len(close)}; highest r = {pairs.r.iloc[0]:.3f}, "
     f"and the top pairs differ in age/sex, so they are different people")

# ---------------------------------------------------------------- low-expression filter
sep_ids = pat.index[sep]
mean_sep = genes[sep_ids].mean(axis=1)
expressed = mean_sep[mean_sep >= 3].index
note("genes", "expressed-gene set for DE/ML", f"mean >= 3 in sepsis: {len(expressed)} of {len(genes)} genes kept "
     f"(full table kept for PCA/clustering)")

# ---------------------------------------------------------------- save
order = ["title", "group", "gender", "male", "age", "hospital", "endotype_cohort", "endotype_class",
         "mortality_event_28days", "time_to_event_28days", "infection_source", "pneumonia_diagnoses",
         "abdominal_sepsis_and_controls", "diabetes_mellitus", "diabetes", "thrombocytopenia",
         "thrombocytopenia_score", "icu_acquired_infection", "icu_acquired_infection_paired",
         "run_date", "array_well", "flag_censored_early", "flag_death_day0", "flag_expr_outlier",
         "flag_sex_mismatch", "expr_sex", "qc_corr_to_median", "sex_score", "max_corr_other_sample"]
pat = pat[order]
pat.to_csv(os.path.join(OUT, "samples_clean.csv"))
genes.round(5).to_csv(os.path.join(OUT, "genes_x_samples.csv"))
genes[sep_ids].round(5).to_csv(os.path.join(OUT, "genes_x_sepsis479.csv"))
genes.loc[expressed, sep_ids].round(5).to_csv(os.path.join(OUT, "genes_x_sepsis479_expressed.csv"))
pairs.head(50).to_csv(os.path.join(OUT, "qc_most_similar_pairs.csv"), index=False)
pd.DataFrame(log).to_csv(os.path.join(OUT, "cleaning_log.csv"), index=False)

print("\nsaved:", sorted(os.listdir(OUT)))
print("group counts:", pat.group.value_counts().to_dict())
print("sepsis deaths:", int(pat.loc[sep, "mortality_event_28days"].sum()), "of", int(sep.sum()))
