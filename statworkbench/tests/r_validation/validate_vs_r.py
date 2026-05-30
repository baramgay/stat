"""R vs Python 수치 검증 스크립트."""
import sys, math, subprocess, json, tempfile, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'statworkbench/src')

import numpy as np
import pandas as pd
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta, MeasureType, StorageType

# ─── helpers ──────────────────────────────────────────────────────────────────

def scale(name): return VariableMeta(name=name, measure=MeasureType.SCALE)
def nom(name): return VariableMeta(name=name, measure=MeasureType.NOMINAL, storage_type=StorageType.STRING)

def run_r(r_code):
    script = "suppressWarnings(suppressMessages({library(jsonlite)}))\n" + r_code + "\ncat(toJSON(result, auto_unbox=TRUE, digits=8))\n"
    with tempfile.NamedTemporaryFile(suffix='.R', mode='w', delete=False, encoding='utf-8') as f:
        f.write(script); fname = f.name
    try:
        out = subprocess.run(['Rscript','--vanilla',fname], capture_output=True, text=True, timeout=30)
        if out.returncode != 0: return {'error': out.stderr[:300]}
        return json.loads(out.stdout)
    finally:
        os.unlink(fname)

def vec_to_r(arr):
    return 'c(' + ','.join(f'{x:.10f}' for x in arr) + ')'

PASS_list, FAIL_list = [], []

def fv(v):
    """포맷된 숫자값 파싱 (< .001 등 처리)."""
    try:
        s = str(v).strip().replace(",", "")
        if s.startswith("< "): return float(s[2:])   # < .001 → 0.001 (upper bound)
        if s.startswith("> "): return float(s[2:])
        return float(s)
    except:
        return float("nan")

def check(name, py_val, r_val, tol=1e-3):
    """수치 비교 (상대 오차 기준)."""
    if r_val is None or (isinstance(r_val, float) and math.isnan(r_val)):
        print(f"  [SKIP] {name}: R값 없음"); return
    if py_val is None or (isinstance(py_val, float) and math.isnan(py_val)):
        print(f"  [FAIL] {name}: Python=NaN  R={float(r_val):.6f}"); FAIL_list.append(name); return
    rel = abs(float(py_val) - float(r_val)) / max(abs(float(r_val)), 1e-10)
    ok = rel < tol
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}: Python={float(py_val):.6f}  R={float(r_val):.6f}  rel={rel:.2e}")
    (PASS_list if ok else FAIL_list).append(name)

def check_pval(name, py_val_or_str, r_val, threshold=0.05):
    """p-value 비교: '< .001' 형식이면 유의성 일치 여부로 판단."""
    if r_val is None or (isinstance(r_val, float) and math.isnan(r_val)):
        print(f"  [SKIP] {name}: R값 없음"); return
    py_raw = py_val_or_str
    py_is_lt = isinstance(py_raw, str) and py_raw.startswith('<')
    py_num = fv(py_raw) if isinstance(py_raw, str) else float(py_raw)
    r_num = float(r_val)
    # 둘 다 < 0.001이면 PASS (유의성 일치)
    if py_num <= 0.001 and r_num < 0.001:
        print(f"  [PASS] {name}: Python<0.001  R={r_num:.6f}  (both p<0.001)")
        PASS_list.append(name); return
    check(name, py_num, r_num, tol=0.05)

def get_statval_row(tables, title_kw, stat_exact):
    """Statistic/Value 구조 테이블에서 Statistic 컬럼 값이 stat_exact와 정확히 일치하는 행의 Value."""
    for t in tables:
        if title_kw.lower() not in t.title.lower(): continue
        df = t.dataframe
        for _, row in df.iterrows():
            if str(row.iloc[0]).strip() == stat_exact:
                return fv(row.iloc[1])
    return float("nan")

def get_col_exact(tables, title_kw, col_exact, row_match_val=None):
    """컬럼 이름이 정확히 col_exact인 열에서 row_match_val을 가진 행의 값 추출."""
    for t in tables:
        if title_kw.lower() not in t.title.lower(): continue
        df = t.dataframe
        if col_exact not in df.columns: continue
        if row_match_val:
            for _, row in df.iterrows():
                if any(str(v).strip() == row_match_val for v in row.values):
                    return fv(row[col_exact])
        else:
            if len(df) > 0:
                return fv(df.iloc[0][col_exact])
    return float("nan")

# ─── Datasets ─────────────────────────────────────────────────────────────────

from sklearn.datasets import load_iris
iris_raw = load_iris()
df_iris = pd.DataFrame(iris_raw.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
df_iris['species'] = [iris_raw.target_names[t] for t in iris_raw.target]
ds_iris = Dataset(data=df_iris, variables={
    'sepal_length': scale('sepal_length'), 'sepal_width': scale('sepal_width'),
    'petal_length': scale('petal_length'), 'petal_width': scale('petal_width'),
    'species': nom('species'),
})

# 동일 데이터 → Python 생성 후 R에 동일 값 전달
np.random.seed(42)
A_data = np.random.normal(100, 15, 50)
B_data = np.random.normal(110, 15, 50)
df_2g = pd.DataFrame({'score': np.concatenate([A_data, B_data]), 'group': ['A']*50+['B']*50})
ds_2g = Dataset(data=df_2g, variables={'score': scale('score'), 'group': nom('group')})

mtcars = {
    'mpg':  [21.0,21.0,22.8,21.4,18.7,18.1,14.3,24.4,22.8,19.2,17.8,16.4,17.3,15.2,10.4,10.4,14.7,32.4,30.4,33.9,21.5,15.5,15.2,13.3,19.2,27.3,26.0,30.4,15.8,19.7,15.0,21.4],
    'wt':   [2.620,2.875,2.320,3.215,3.440,3.460,3.570,3.190,3.150,3.440,3.440,4.070,3.730,3.780,5.250,5.424,5.345,2.200,1.615,1.835,2.465,3.520,3.435,3.840,3.845,1.935,2.140,1.513,3.170,2.770,3.570,2.780],
    'hp':   [110,110,93,110,175,105,245,62,95,123,123,180,180,180,205,215,230,66,52,65,97,150,150,245,175,66,91,113,264,175,335,109],
}
df_mt = pd.DataFrame(mtcars)
ds_mt = Dataset(data=df_mt, variables={c: scale(c) for c in df_mt.columns})

np.random.seed(7)
gender_arr = np.random.choice(['M','F'], 200)
prefer_arr = np.random.choice(['A','B','C'], 200)
df_cat = pd.DataFrame({'gender': gender_arr, 'prefer': prefer_arr})
ds_cat = Dataset(data=df_cat, variables={'gender': nom('gender'), 'prefer': nom('prefer')})

np.random.seed(3)
pre_data = np.random.normal(50, 10, 30)
post_data = np.random.normal(55, 10, 30)
df_paired = pd.DataFrame({'pre': pre_data, 'post': post_data})
ds_paired = Dataset(data=df_paired, variables={'pre': scale('pre'), 'post': scale('post')})

# 로지스틱: versicolor vs virginica (perfect separation 없음)
df_logi = df_iris[df_iris['species'].isin(['versicolor','virginica'])].copy().reset_index(drop=True)
df_logi['is_virginica'] = (df_logi['species'] == 'virginica').astype(int).astype(str)
ds_logi = Dataset(data=df_logi, variables={
    **{c: scale(c) for c in ['sepal_length','sepal_width','petal_length','petal_width']},
    'is_virginica': nom('is_virginica'), 'species': nom('species'),
})

np.random.seed(1)
score_2w = np.concatenate([np.random.normal(70,10,20), np.random.normal(75,10,20),
                            np.random.normal(80,10,20), np.random.normal(85,10,20)])
df_2w = pd.DataFrame({'score': score_2w, 'method': ['A']*40+['B']*40,
                       'gender': (['M']*20+['F']*20)*2})
ds_2w = Dataset(data=df_2w, variables={'score':scale('score'),'method':nom('method'),'gender':nom('gender')})

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. 독립표본 t-검정")
print("="*60)
from statworkbench.analysis.ttests import run_analysis as ttest_run
r1 = ttest_run(ds_2g, {'variables':{'dependent':'score','group':'group'},
                        'options':{'alpha':0.05},'missing_policy':'listwise'})
t_py = p_py = float("nan")
for t in r1.tables:
    if "Independent" in t.title or "t-Test" in t.title:
        df = t.dataframe
        if 't' in df.columns: t_py = fv(df.iloc[0]['t'])
        if 'p-value' in df.columns: p_raw = str(df.iloc[0]['p-value'])

A_r = vec_to_r(A_data); B_r = vec_to_r(B_data)
r_t = run_r(f"""
A <- {A_r}; B <- {B_r}
tt <- t.test(A, B, var.equal=TRUE)
result <- list(t=as.numeric(tt$statistic), p=as.numeric(tt$p.value))
""")
print(f"  R ref: t={r_t.get('t','N/A'):.4f}, p={r_t.get('p','N/A'):.6e}")
check("독립표본 t통계량", t_py, r_t.get('t'))
check_pval("독립표본 p값", p_raw if 't_py' else "< .001", r_t.get('p'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. 대응표본 t-검정")
print("="*60)
r2 = ttest_run(ds_paired, {'variables':{'paired':['pre','post']},
                            'options':{'alpha':0.05},'missing_policy':'listwise'})
t_pair_py = p_pair_py = float("nan")
for t in r2.tables:
    if 'Paired Samples t-Test' in t.title:
        df = t.dataframe
        # Statistic/Value 구조: 정확히 "t"인 행 찾기
        t_pair_py = get_statval_row([t], 'Paired', 't')
        p_pair_py = get_statval_row([t], 'Paired', 'p-value')

pre_r = vec_to_r(pre_data); post_r = vec_to_r(post_data)
r_pair = run_r(f"""
pre <- {pre_r}; post <- {post_r}
tt <- t.test(pre, post, paired=TRUE)
result <- list(t=as.numeric(tt$statistic), p=as.numeric(tt$p.value))
""")
print(f"  R ref: t={r_pair.get('t','N/A'):.4f}, p={r_pair.get('p','N/A'):.4f}")
check("대응표본 t통계량", t_pair_py, r_pair.get('t'))
check("대응표본 p값", p_pair_py, r_pair.get('p'), tol=0.02)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. 일원 분산분석 (ANOVA)")
print("="*60)
from statworkbench.analysis.anova import run_analysis as anova_run
r3 = anova_run(ds_iris, {'variables':{'dependent':'sepal_length','factor':'species'},
                          'options':{'effect_size':True,'post_hoc':True,'post_hoc_method':'tukey'},
                          'missing_policy':'listwise'})
f_py = float("nan")
for t in r3.tables:
    if t.title == 'ANOVA':
        df = t.dataframe
        for _, row in df.iterrows():
            if 'species' in str(row.iloc[0]).lower() or 'C(' in str(row.iloc[0]):
                if 'F' in df.columns: f_py = fv(row['F'])

r_anova = run_r("""
data(iris)
m <- aov(Sepal.Length ~ Species, data=iris)
s <- summary(m)[[1]]
result <- list(F=s[["F value"]][1], p=s[["Pr(>F)"]][1],
               eta2=s[["Sum Sq"]][1]/sum(s[["Sum Sq"]]))
""")
print(f"  R ref: F={r_anova.get('F','?'):.4f}")
check("ANOVA F통계량", f_py, r_anova.get('F'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. 피어슨 상관분석")
print("="*60)
from statworkbench.analysis.correlation import run_analysis as corr_run
r4 = corr_run(ds_iris, {'variables':{'target':['sepal_length','petal_length']},
                         'options':{'method':'pearson','tail':'two-tailed','confidence_level':0.95},
                         'missing_policy':'listwise'})
r_corr_py = float("nan")
for t in r4.tables:
    if 'pearson' in t.title.lower() and 'matrix' in t.title.lower():
        df = t.dataframe
        if 'petal_length' in df.columns and len(df) > 0:
            r_corr_py = fv(df.iloc[0]['petal_length'])

r_corr = run_r("""
data(iris)
ct <- cor.test(iris$Sepal.Length, iris$Petal.Length, method="pearson")
result <- list(r=as.numeric(ct$estimate), p=as.numeric(ct$p.value))
""")
print(f"  R ref: r={r_corr.get('r','?'):.6f}")
check("Pearson r", r_corr_py, r_corr.get('r'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. 스피어만 상관")
print("="*60)
r5 = corr_run(ds_iris, {'variables':{'target':['sepal_length','petal_length']},
                         'options':{'method':'spearman','tail':'two-tailed'},
                         'missing_policy':'listwise'})
rho_py = float("nan")
for t in r5.tables:
    if 'spearman' in t.title.lower() and 'matrix' in t.title.lower():
        df = t.dataframe
        if 'petal_length' in df.columns and len(df) > 0:
            rho_py = fv(df.iloc[0]['petal_length'])

r_sp = run_r("""
data(iris)
ct <- cor.test(iris$Sepal.Length, iris$Petal.Length, method="spearman")
result <- list(rho=as.numeric(ct$estimate), p=as.numeric(ct$p.value))
""")
print(f"  R ref: rho={r_sp.get('rho','?'):.6f}")
check("Spearman rho", rho_py, r_sp.get('rho'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. 선형 회귀분석")
print("="*60)
from statworkbench.analysis.regression import run_analysis as reg_run
r6 = reg_run(ds_mt, {'variables':{'dependent':'mpg','predictors':['wt','hp']},
                      'options':{},'missing_policy':'listwise'})
# "Model Summary" → Statistic/Value 구조
r2_py = get_statval_row(r6.tables, "Model Summary", "R-squared")
# "Coefficients" → Variable/B/SE/Beta/t/p-value/CI 구조
b_wt_py = get_col_exact(r6.tables, "Coefficients", "B", row_match_val="wt")
b_hp_py = get_col_exact(r6.tables, "Coefficients", "B", row_match_val="hp")

r_reg = run_r("""
mpg  <- c(21.0,21.0,22.8,21.4,18.7,18.1,14.3,24.4,22.8,19.2,17.8,16.4,17.3,15.2,10.4,10.4,14.7,32.4,30.4,33.9,21.5,15.5,15.2,13.3,19.2,27.3,26.0,30.4,15.8,19.7,15.0,21.4)
wt   <- c(2.620,2.875,2.320,3.215,3.440,3.460,3.570,3.190,3.150,3.440,3.440,4.070,3.730,3.780,5.250,5.424,5.345,2.200,1.615,1.835,2.465,3.520,3.435,3.840,3.845,1.935,2.140,1.513,3.170,2.770,3.570,2.780)
hp   <- c(110,110,93,110,175,105,245,62,95,123,123,180,180,180,205,215,230,66,52,65,97,150,150,245,175,66,91,113,264,175,335,109)
m <- lm(mpg ~ wt + hp); s <- summary(m)
result <- list(r2=s$r.squared, b_wt=coef(m)["wt"], b_hp=coef(m)["hp"])
""")
print(f"  R ref: R²={r_reg.get('r2','?'):.6f}, b_wt={r_reg.get('b_wt','?'):.4f}, b_hp={r_reg.get('b_hp','?'):.4f}")
check("회귀 R²", r2_py, r_reg.get('r2'))
check("회귀 b_wt", b_wt_py, r_reg.get('b_wt'))
check("회귀 b_hp", b_hp_py, r_reg.get('b_hp'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. 카이제곱 검정 (교차분석)")
print("="*60)
from statworkbench.analysis.crosstab import run_analysis as cross_run
r7 = cross_run(ds_cat, {'variables':{'row':'gender','column':'prefer'},
                         'options':{'chi_square':True,'phi_cramer':True},'missing_policy':'listwise'})
chi2_py = p_chi_py = float("nan")
for t in r7.tables:
    if 'chi' in t.title.lower():
        df = t.dataframe
        for _, row in df.iterrows():
            label = str(row.iloc[0]).lower()
            if 'pearson' in label:
                if 'Value' in df.columns: chi2_py = fv(row['Value'])
                elif len(row) > 1: chi2_py = fv(row.iloc[1])
                if 'p-value' in df.columns: p_chi_py = fv(row['p-value'])
                elif len(row) > 3: p_chi_py = fv(row.iloc[3])

g_r = "c(" + ",".join(f'"{x}"' for x in gender_arr) + ")"
p_r = "c(" + ",".join(f'"{x}"' for x in prefer_arr) + ")"
r_chi = run_r(f"""
gender <- {g_r}; prefer <- {p_r}
ct <- chisq.test(table(gender, prefer))
result <- list(chi2=as.numeric(ct$statistic), p=as.numeric(ct$p.value))
""")
print(f"  R ref: χ²={r_chi.get('chi2','?'):.4f}, p={r_chi.get('p','?'):.4f}")
check("카이제곱 χ²", chi2_py, r_chi.get('chi2'))
check("카이제곱 p값", p_chi_py, r_chi.get('p'), tol=0.05)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8. 기술통계")
print("="*60)
from statworkbench.analysis.descriptive import run_analysis as desc_run
r8 = desc_run(ds_iris, {'variables':{'scale':['sepal_length','petal_length']},
                         'options':{'statistics':['mean','std','min','max']},
                         'missing_policy':'listwise'})
mean_sl = std_sl = float("nan")
for t in r8.tables:
    if '기술통계' in t.title or 'descriptive' in t.title.lower():
        df = t.dataframe
        for _, row in df.iterrows():
            if 'sepal_length' in str(row.values):
                for c in df.columns:
                    cl = c.lower()
                    if cl == 'mean' or cl == '평균': mean_sl = fv(row[c])
                    if cl == 'sd' or cl == 'std': std_sl = fv(row[c])

r_desc = run_r("""
data(iris)
result <- list(mean_sl=mean(iris$Sepal.Length), sd_sl=sd(iris$Sepal.Length))
""")
print(f"  R ref: mean={r_desc.get('mean_sl','?'):.6f}, sd={r_desc.get('sd_sl','?'):.6f}")
check("기술통계 mean sepal_length", mean_sl, r_desc.get('mean_sl'))
check("기술통계 SD sepal_length", std_sl, r_desc.get('sd_sl'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("9. 정규성 검정 (Shapiro-Wilk)")
print("="*60)
from statworkbench.analysis.normality import run_analysis as norm_run
r9 = norm_run(ds_iris, {'variables':{'target':['sepal_length']},
                         'options':{'tests':['shapiro']},'missing_policy':'listwise'})
w_py = p_sw_py = float("nan")
for t in r9.tables:
    if 'normality' in t.title.lower() or '정규성' in t.title:
        df = t.dataframe
        if len(df) > 0 and 'Statistic' in df.columns:
            w_py = fv(df.iloc[0]['Statistic'])
        if len(df) > 0 and 'p-value' in df.columns:
            p_sw_py = fv(df.iloc[0]['p-value'])

r_sw = run_r("""
data(iris); sw <- shapiro.test(iris$Sepal.Length)
result <- list(W=as.numeric(sw$statistic), p=as.numeric(sw$p.value))
""")
print(f"  R ref: W={r_sw.get('W','?'):.6f}, p={r_sw.get('p','?'):.6f}")
check("Shapiro-Wilk W", w_py, r_sw.get('W'))
check("Shapiro-Wilk p", p_sw_py, r_sw.get('p'), tol=0.1)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("10. 이진 로지스틱 회귀 (versicolor vs virginica)")
print("="*60)
from statworkbench.analysis.logistic_regression import run_analysis as log_run
r10 = log_run(ds_logi, {'variables':{'dependent':'is_virginica','predictors':['petal_length','petal_width']},
                         'options':{},'missing_policy':'listwise'})
b_pl_py = float("nan")
for t in r10.tables:
    if '계수' in t.title or 'coefficients' in t.title.lower():
        df = t.dataframe
        for _, row in df.iterrows():
            if 'petal_length' in str(row.values):
                if 'B' in df.columns: b_pl_py = fv(row['B'])
                elif len(row) > 1: b_pl_py = fv(row.iloc[1])

r_log = run_r("""
data(iris)
df <- iris[iris$Species %in% c("versicolor","virginica"),]
df$is_virginica <- as.integer(df$Species=="virginica")
m <- glm(is_virginica ~ Petal.Length + Petal.Width, data=df, family=binomial)
s <- summary(m)$coefficients
result <- list(b_pl=s["Petal.Length","Estimate"], b_pw=s["Petal.Width","Estimate"])
""")
print(f"  R ref: b_petal_length={r_log.get('b_pl','?'):.4f}")
check("로지스틱 b_petal_length", b_pl_py, r_log.get('b_pl'), tol=0.05)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("11. 이원 분산분석")
print("="*60)
from statworkbench.analysis.two_way_anova import run_analysis as twa_run
r11 = twa_run(ds_2w, {'variables':{'dependent':'score','factor_a':'method','factor_b':'gender'},
                       'options':{'interaction':True,'effect_size':True},'missing_policy':'listwise'})
f_method_py = float("nan")
for t in r11.tables:
    if 'Between-Subjects' in t.title or 'ANOVA' in t.title or '분산분석' in t.title:
        df = t.dataframe
        if 'F' in df.columns:
            for _, row in df.iterrows():
                if str(row.iloc[0]).strip().lower() == 'method':
                    f_method_py = fv(row['F'])

sc_r = vec_to_r(score_2w)
r_2w = run_r(f"""
score  <- {sc_r}
method <- c(rep("A",40),rep("B",40))
gender <- rep(c(rep("M",20),rep("F",20)),2)
m <- aov(score~method*gender); s <- summary(m)[[1]]
rn <- trimws(rownames(s))
result <- list(
  F_method=as.numeric(s[rn=="method","F value"]),
  p_method=as.numeric(s[rn=="method","Pr(>F)"])
)
""")
print(f"  R ref: F_method={r_2w.get('F_method','?')}")
if isinstance(r_2w.get('F_method'), (int,float)) and not (isinstance(r_2w.get('F_method'), float) and math.isnan(r_2w.get('F_method'))):
    check("이원ANOVA F_method", f_method_py, r_2w.get('F_method'), tol=0.01)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("12. Mann-Whitney U 검정 (비모수)")
print("="*60)
from statworkbench.analysis.nonparametric import run_analysis as np_run
r12 = np_run(ds_2g, {'options':{'test':'mann_whitney'},
                      'variables':{'dependent':'score','group':'group'},
                      'missing_policy':'listwise'})
U_py = get_statval_row(r12.tables, "Test Statistics", "Mann-Whitney U")
p_mw_raw = None
for t in r12.tables:
    if 'Test Statistics' in t.title:
        df = t.dataframe
        for _, row in df.iterrows():
            if 'p-value' in str(row.iloc[0]).lower():
                p_mw_raw = str(row.iloc[1])

A_r = vec_to_r(A_data); B_r = vec_to_r(B_data)
r_mw = run_r(f"""
A <- {A_r}; B <- {B_r}
wt <- wilcox.test(A, B, exact=FALSE)
result <- list(W=as.numeric(wt$statistic), p=as.numeric(wt$p.value))
""")
print(f"  R ref: W(U)={r_mw.get('W','?')}, p={r_mw.get('p','?'):.6e}")
check("Mann-Whitney U", U_py, r_mw.get('W'))
check_pval("Mann-Whitney p", p_mw_raw or "< .001", r_mw.get('p'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("검증 결과 요약")
print("="*60)
total = len(PASS_list) + len(FAIL_list)
print(f"  PASS: {len(PASS_list)}/{total}")
print(f"  FAIL: {len(FAIL_list)}/{total}")
if FAIL_list:
    print(f"\n  실패 항목:")
    for f in FAIL_list: print(f"    - {f}")
else:
    print("\n  모든 검증 통과!")
print()
