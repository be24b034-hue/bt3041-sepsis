"""Exploratory pattern analysis on the cleaned data (run 01_clean.py first).
Writes tables to results/ and figures to figures/; prints a summary."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
CL, RES, FIG = (os.path.join(HERE, d) for d in ("cleaned", "results", "figures"))
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
PAL = {"Healthy": "#2a9d8f", "Other_ICU": "#8d99ae", "Sepsis": "#e63946"}
MARS = {"Mars1": "#264653", "Mars2": "#e76f51", "Mars3": "#2a9d8f", "Mars4": "#e9c46a"}

pat = pd.read_csv(os.path.join(CL, "samples_clean.csv"), index_col=0, parse_dates=["run_date"])
genes = pd.read_csv(os.path.join(CL, "genes_x_samples.csv"), index_col=0)
E = pd.read_csv(os.path.join(CL, "genes_x_sepsis479_expressed.csv"), index_col=0)
sep = pat[pat.group == "Sepsis"].copy()
sep["died"] = sep.mortality_event_28days.astype(int)
sep["outcome"] = sep.died.map({0: "Survived", 1: "Died"})

# ================================================================ 1. cohort table
def describe(d):
    return pd.Series({"n": len(d), "age_median": d.age.median(),
                      "age_IQR": f"{d.age.quantile(.25):.0f}-{d.age.quantile(.75):.0f}",
                      "male_%": round(100 * d.male.mean(), 1)})
table1 = pat.groupby("group")[["age", "male"]].apply(describe)
table1.to_csv(os.path.join(RES, "table1_cohort.csv"))
print("=== Table 1\n", table1, sep="")
print(f"Sepsis 28-day deaths: {sep.died.sum()} / {len(sep)} ({100*sep.died.mean():.1f}%)")

# ================================================================ 2. what relates to death?
rows = []
def add(var, test, stat, p, detail):
    rows.append({"variable": var, "test": test, "statistic": round(stat, 3), "p": p, "detail": detail})

a, b = sep.age[sep.died == 1], sep.age[sep.died == 0]
add("age", "Welch t", stats.ttest_ind(a, b, equal_var=False)[0], stats.ttest_ind(a, b, equal_var=False)[1],
    f"mean died {a.mean():.1f} vs survived {b.mean():.1f}")
add("age", "Mann-Whitney U", stats.mannwhitneyu(a, b)[0], stats.mannwhitneyu(a, b)[1],
    f"median died {a.median():.0f} vs survived {b.median():.0f}")

def cat_test(var, data=sep, label=None):
    ct = pd.crosstab(data[var], data.died)
    rate = (ct[1] / ct.sum(axis=1) * 100).round(1).to_dict()
    if ct.shape == (2, 2):
        o, p = stats.fisher_exact(ct)
        add(label or var, "Fisher exact", o, p, f"death % {rate}; n={int(ct.values.sum())}")
    else:
        c, p, _, exp = stats.chi2_contingency(ct)
        add(label or var, "chi-square", c, p, f"death % {rate}; n={int(ct.values.sum())}; min expected {exp.min():.1f}")

cat_test("gender", label="sex")
cat_test("diabetes_mellitus", label="diabetes")
cat_test("hospital")
cat_test("endotype_class", label="Mars endotype")
cat_test("infection_source", data=sep[sep.infection_source != "pneumonia+abdominal"])
cat_test("icu_acquired_infection", data=sep.dropna(subset=["icu_acquired_infection"]), label="ICU-acquired infection")
t = sep.dropna(subset=["thrombocytopenia_score"])
add("platelet score (0 very low - 3 normal)", "Mann-Whitney U",
    *stats.mannwhitneyu(t.thrombocytopenia_score[t.died == 1], t.thrombocytopenia_score[t.died == 0]),
    f"n={len(t)}; median died {t.thrombocytopenia_score[t.died==1].median()} vs survived {t.thrombocytopenia_score[t.died==0].median()}")
sep["run_date_s"] = sep.run_date.dt.strftime("%Y-%m-%d")
cat_test("run_date_s", label="array run date (batch)")

assoc = pd.DataFrame(rows)
assoc["p"] = assoc.p.map(lambda v: float(f"{v:.3g}"))
assoc.to_csv(os.path.join(RES, "mortality_associations.csv"), index=False)
print("\n=== What is associated with 28-day death (sepsis, n=479)")
print(assoc.to_string(index=False))

# age gradient
sep["age_band"] = pd.cut(sep.age, [0, 50, 60, 70, 80, 100], labels=["<50", "50-59", "60-69", "70-79", "80+"])
age_rate = sep.groupby("age_band", observed=True).died.agg(["size", "mean"]).round(3)
print("\ndeath rate by age band:\n", age_rate.to_string(), sep="")

# multivariable logistic regression
sep["Mars1"] = (sep.endotype_class == "Mars1").astype(int)
m = smf.logit("died ~ age + male + Mars1 + C(hospital)", data=sep).fit(disp=0)
orr = pd.DataFrame({"OR": np.exp(m.params), "CI_low": np.exp(m.conf_int()[0]), "CI_high": np.exp(m.conf_int()[1]),
                    "p": m.pvalues}).drop("Intercept").round(3)
orr.to_csv(os.path.join(RES, "logistic_age_sex_mars1_hospital.csv"))
print("\nlogistic regression died ~ age + sex + Mars1 + hospital:\n", orr.to_string(), sep="")

# ================================================================ 3. genes vs death
def de(E, mask):
    t_, p_ = stats.ttest_ind(E.loc[:, mask], E.loc[:, ~mask], axis=1, equal_var=False)
    r = pd.DataFrame({"log2FC": E.loc[:, mask].mean(axis=1) - E.loc[:, ~mask].mean(axis=1), "t": t_, "p": p_}, index=E.index)
    r["FDR"] = multipletests(r.p, method="fdr_bh")[1]
    return r.sort_values("p")

died_mask = (sep.loc[E.columns, "died"] == 1).values
de_death = de(E, died_mask)
de_death.to_csv(os.path.join(RES, "DE_died_vs_survived.csv"))
sig = de_death[de_death.FDR < 0.05]
print(f"\n=== Genes vs death: {len(sig)} genes FDR<0.05 of {len(E)} "
      f"({(sig.log2FC < 0).sum()} lower, {(sig.log2FC > 0).sum()} higher in non-survivors); "
      f"max |log2FC| {sig.log2FC.abs().max():.2f}")
print("top lower in died:", sig[sig.log2FC < 0].head(8).index.tolist())
print("top higher in died:", sig[sig.log2FC > 0].head(8).index.tolist())

# sensitivity: drop flagged samples (sex mismatch, expression outliers, early censoring)
flag = (sep.flag_sex_mismatch | sep.flag_expr_outlier | sep.flag_censored_early).reindex(E.columns).values
de_s = de(E.loc[:, ~flag], died_mask[~flag])
fl = sep.loc[E.columns[flag]]
print(f"sensitivity without {flag.sum()} flagged sepsis samples ({fl.died.sum()} died, "
      f"{(fl.endotype_class == 'Mars1').sum()} Mars1): {(de_s.FDR < 0.05).sum()} genes FDR<0.05; "
      f"overlap with main list {len(set(sig.index) & set(de_s.index[de_s.FDR < 0.05]))}; "
      f"t-statistics correlate r = {np.corrcoef(de_death.t, de_s.t.reindex(de_death.index))[0, 1]:.3f}")

# sepsis vs healthy
H = genes.loc[E.index, pat.index[pat.group == "Healthy"]]
SH = pd.concat([E, H], axis=1)
de_sh = de(SH, np.r_[np.ones(E.shape[1], bool), np.zeros(H.shape[1], bool)])
de_sh.to_csv(os.path.join(RES, "DE_sepsis_vs_healthy.csv"))
s2 = de_sh[(de_sh.FDR < 0.05) & (de_sh.log2FC.abs() > 1)]
print(f"\n=== Sepsis vs healthy: {(de_sh.FDR < 0.05).sum()} genes FDR<0.05, {len(s2)} also >2-fold")
print("top up in sepsis:", s2[s2.log2FC > 0].sort_values("log2FC", ascending=False).head(8).index.tolist())
print("top down in sepsis:", s2[s2.log2FC < 0].sort_values("log2FC").head(8).index.tolist())

# genes tracking age within sepsis
rho = E.T.corrwith(sep.loc[E.columns, "age"], method="spearman").sort_values()
print("\ngenes most correlated with age (Spearman):", rho.head(4).round(2).to_dict(), rho.tail(4).round(2).to_dict())

# ================================================================ 4. structure: PCA
def pca(df, n=10):
    top = df.var(axis=1).sort_values(ascending=False).index[:2000]
    p = PCA(n_components=n, random_state=0)
    return p.fit_transform(StandardScaler().fit_transform(df.loc[top].T)), p.explained_variance_ratio_

P_all, v_all = pca(genes)
P_sep, v_sep = pca(genes[sep.index])
sep["PC1"], sep["PC2"] = P_sep[:, 0], P_sep[:, 1]
print(f"\n=== PCA: all samples PC1+PC2 = {100*v_all[:2].sum():.0f}% variance; sepsis only = {100*v_sep[:2].sum():.0f}%")
for pc in ["PC1", "PC2"]:
    print(f"{pc}: rho with age {stats.spearmanr(sep[pc], sep.age)[0]:.2f}; "
          f"died vs survived Mann-Whitney p = {stats.mannwhitneyu(sep[pc][sep.died==1], sep[pc][sep.died==0])[1]:.3g}")
r2 = [smf.ols("y ~ C(d)", data=pd.DataFrame({"y": P_sep[:, i], "d": sep.run_date_s.values})).fit().rsquared for i in range(5)]
print("batch: R^2 of run date on sepsis PC1-5:", np.round(r2, 3).tolist())

# ================================================================ 5. figures
# Fig 1: QC
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for g, col in PAL.items():
    for s in pat.index[pat.group == g][:25]:
        sns.kdeplot(genes[s], ax=ax[0], color=col, lw=0.6, alpha=0.5)
ax[0].set(title="A  Expression density (25 samples/group)", xlabel="log2 expression")
ax[0].legend(handles=[plt.Line2D([], [], color=c, label=g) for g, c in PAL.items()], frameon=False)
o = pat.sort_values("qc_corr_to_median")
ax[1].scatter(range(len(o)), o.qc_corr_to_median, s=6, c=np.where(o.flag_expr_outlier, "#e63946", "#8d99ae"))
ax[1].set(title=f"B  Correlation with median sample ({int(pat.flag_expr_outlier.sum())} flagged, kept)",
          xlabel="samples (sorted)", ylabel="Pearson r")
gx = genes.loc[["XIST", "RPS4Y1"]].T.join(pat[["gender", "flag_sex_mismatch"]])
for sx, col in [("female", "#e76f51"), ("male", "#264653")]:
    d = gx[gx.gender == sx]
    ax[2].scatter(d.XIST, d.RPS4Y1, s=8, c=col, alpha=0.6, label=f"recorded {sx}")
mm = gx[gx.flag_sex_mismatch]
ax[2].scatter(mm.XIST, mm.RPS4Y1, s=70, facecolors="none", edgecolors="red", lw=1.5, label=f"mismatch (n={len(mm)})")
ax[2].axvline(3.5, ls=":", c="grey"); ax[2].axhline(5, ls=":", c="grey")
ax[2].set(title="C  Sex check: XIST (X) vs RPS4Y1 (Y)", xlabel="XIST", ylabel="RPS4Y1"); ax[2].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_qc.png"), dpi=200); plt.close()

# Fig 2: cohort and mortality patterns
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
sns.boxplot(data=pat, x="group", y="age", order=list(PAL), palette=PAL, ax=ax[0], hue="group", legend=False)
ax[0].set(title="A  Age by group", xlabel="")
sns.histplot(data=sep, x="age", hue="outcome", bins=25, kde=True, ax=ax[1], palette={"Died": "#e63946", "Survived": "#457b9d"})
ax[1].set(title=f"B  Age by 28-day outcome (MWU p={stats.mannwhitneyu(a, b)[1]:.3f})")
ar = sep.groupby("age_band", observed=True).died.mean() * 100
ax[2].bar(ar.index.astype(str), ar.values, color="#e63946")
for i, (v, n) in enumerate(zip(ar.values, sep.age_band.value_counts().reindex(ar.index).values)):
    ax[2].text(i, v + 0.8, f"{v:.0f}%\nn={n}", ha="center", fontsize=8)
ax[2].set(title="C  Death rate by age band", ylabel="28-day mortality (%)", xlabel="age", ylim=(0, 45))
mr = sep.groupby("endotype_class").died.mean() * 100
ax[3].bar(mr.index, mr.values, color=[MARS[k] for k in mr.index])
for i, (k, v) in enumerate(mr.items()):
    ax[3].text(i, v + 0.8, f"{v:.0f}%\nn={(sep.endotype_class == k).sum()}", ha="center", fontsize=8)
p_mars = assoc.loc[assoc.variable == "Mars endotype", "p"].iloc[0]
ax[3].set(title=f"D  Death rate by Mars endotype (chi2 p={p_mars:.3f})", ylabel="28-day mortality (%)", ylim=(0, 45))
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_cohort_mortality.png"), dpi=200); plt.close()

# Fig 3: PCA
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
d = pd.DataFrame(P_all[:, :2], index=genes.columns, columns=["PC1", "PC2"]).join(pat.group)
sns.scatterplot(data=d, x="PC1", y="PC2", hue="group", palette=PAL, s=10, ax=ax[0], linewidth=0)
ax[0].set(title=f"A  All 802 samples ({100*v_all[0]:.0f}% / {100*v_all[1]:.0f}% variance)")
sns.scatterplot(data=sep, x="PC1", y="PC2", hue="endotype_class", palette=MARS, hue_order=list(MARS), s=12, ax=ax[1], linewidth=0)
ax[1].set(title="B  Sepsis only, by Mars endotype")
sns.scatterplot(data=sep.sort_values("died"), x="PC1", y="PC2", hue="outcome", palette={"Died": "#e63946", "Survived": "#c9d6df"},
                s=12, ax=ax[2], linewidth=0)
ax[2].set(title="C  Sepsis only, by 28-day outcome")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_pca.png"), dpi=200); plt.close()

# Fig 4: genes vs death
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1.3]})
c = np.where(de_death.FDR >= 0.05, "#d9d9d9", np.where(de_death.log2FC > 0, "#e63946", "#457b9d"))
ax[0].scatter(de_death.log2FC, -np.log10(de_death.p), c=c, s=5)
for gname in ["CX3CR1", "TNFSF12", "TNFRSF8", "AMICA1", "TUBG2", "NCRNA00188"]:
    ax[0].annotate(gname, (de_death.log2FC[gname], -np.log10(de_death.p[gname])), fontsize=7,
                   xytext=(4, -3), textcoords="offset points")
ax[0].set(title=f"A  Died vs survived: {len(sig)} genes FDR<0.05", xlabel="log2 fold change (died - survived)", ylabel="-log10 p")
ax[1].hist(de_death.p, bins=40, color="#8d99ae"); ax[1].set(title="B  p-value histogram (spike = real signal)", xlabel="p")
show = sig[sig.log2FC < 0].index[:3].tolist() + sig[sig.log2FC > 0].index[:3].tolist()
long = E.loc[show].T.join(sep.outcome).melt(id_vars="outcome", var_name="gene", value_name="log2 expression")
sns.boxplot(data=long, x="gene", y="log2 expression", hue="outcome", palette={"Died": "#e63946", "Survived": "#457b9d"},
            ax=ax[2], fliersize=1)
ax[2].set(title="C  Top genes lower / higher in non-survivors", xlabel=""); ax[2].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_genes_death.png"), dpi=200); plt.close()

# Fig 5: Kaplan-Meier by Mars (simple product-limit estimator)
def km(time, event):
    ts = np.sort(time[event == 1].unique()); s, out = 1.0, [(0, 1.0)]
    for t_ in ts:
        at_risk = (time >= t_).sum(); d_ = ((time == t_) & (event == 1)).sum()
        s *= 1 - d_ / at_risk; out.append((t_, s))
    out.append((28, s)); return np.array(out)
plt.figure(figsize=(6.5, 4.6))
for k, col in MARS.items():
    dd = sep[sep.endotype_class == k]
    c_ = km(dd.time_to_event_28days, dd.died)
    plt.step(c_[:, 0], c_[:, 1], where="post", color=col, lw=2, label=f"{k} (n={len(dd)}, {100*dd.died.mean():.0f}% died)")
plt.ylim(0.55, 1.01); plt.xlabel("days since ICU admission"); plt.ylabel("fraction alive")
plt.title("Kaplan-Meier survival by Mars endotype"); plt.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_km_mars.png"), dpi=200); plt.close()
print("\nfigures:", sorted(os.listdir(FIG)))
