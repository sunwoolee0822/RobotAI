"""4.3 결측률 vs 라벨 — 블록별 barplot + Mann-Whitney"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15})

D = Path("data/numpy_all_chunk_72_24feat")
OUT = Path("analysis_72"); (OUT/"plots").mkdir(parents=True, exist_ok=True)

GROUPS = {"A": ["hr","rr"], "B1": ["core_temp","skin_temp"],
          "B2": ["bp_sys","bp_dia","glucose"], "C": ["light_sensor"],
          "D": ["spo2","hrv","proximity"],
          "E": ["step","distance","screen_time","wake_time","sleep_time",
                "deep_sleep_time","rem_sleep_time","light_sleep_time","total_sleep_time"],
          "F": ["EMA_Anxiety","EMA_Depression","EMA_Sleep","EMA_Stress"]}
ORDER = ["A","B1","B2","C","D","E","F"]
COLORS = {"A":"#e74c3c","B1":"#5dade2","B2":"#154360","C":"#9b59b6",
          "D":"#d35400","E":"#27ae60","F":"#34495e"}
# 블록 색이 무엇을 뜻하는지 축에 직접 명시 (색 = 블록, 채움 = 그룹)
BLOCK_DESC = {"A":"Cardio-\nrespiratory", "B1":"Body\ntemp.", "B2":"BP /\nglucose",
              "C":"Ambient\nlight", "D":"SpO$_2$ / HRV /\nproximity",
              "E":"Activity &\nsleep", "F":"EMA\nself-report"}

# subject 단위 결측률
subj = pd.read_csv(OUT/"missing_rate_by_subject_feature_72.csv")
subj["subject_id"] = subj["subject_id"].astype(str)
f2g = {f:g for g,fs in GROUPS.items() for f in fs}
subj["block"] = subj["feature"].map(f2g)
blk = subj.groupby(["subject_id","block"])["missing_rate"].mean().reset_index()

# subject 라벨 (윈도우 하나라도 1이면 1)
sids = [str(s) for s in json.loads((D/"subject_ids.json").read_text())]
gt = np.load(D/"gt.npy").ravel().astype(int)
lab = pd.DataFrame({"subject_id": sids, "y": gt}).groupby("subject_id")["y"].max().reset_index()

df = blk.merge(lab, on="subject_id")
n1 = int((lab["y"]==1).sum()); n0 = int((lab["y"]==0).sum())

def star(p):
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."

rows, fig_data = [], []
for b in ORDER:
    d = df[df["block"]==b]
    a = d.loc[d["y"]==0,"missing_rate"].values
    c = d.loc[d["y"]==1,"missing_rate"].values
    u,p = mannwhitneyu(c,a,alternative="two-sided")
    rows.append({"block":b,"n_control":len(a),"n_patient":len(c),
                 "mean_control":a.mean(),"sem_control":a.std(ddof=1)/np.sqrt(len(a)),
                 "mean_patient":c.mean(),"sem_patient":c.std(ddof=1)/np.sqrt(len(c)),
                 "U":u,"p":p,"sig":star(p)})
    fig_data.append((b,a,c,star(p)))
tab = pd.DataFrame(rows)
tab.to_csv(OUT/"missing_vs_label_block.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(13,7))
x = np.arange(len(ORDER)); w = 0.34
# 색 = 피처 블록, 채움(연한+빗금 vs 진한 단색) = 그룹.
# 그룹 구분을 명도와 해치로 이중 인코딩해 색맹·흑백 인쇄에서도 읽히게 함.
for i,(b,a,c,s) in enumerate(fig_data):
    ax.bar(x[i]-w/2-.01, a.mean(), w, yerr=a.std(ddof=1)/np.sqrt(len(a)), capsize=4,
           color=COLORS[b], alpha=.35, edgecolor=COLORS[b], hatch="//", linewidth=1.2,
           error_kw={"ecolor":"#333333"})
    ax.bar(x[i]+w/2+.01, c.mean(), w, yerr=c.std(ddof=1)/np.sqrt(len(c)), capsize=4,
           color=COLORS[b], alpha=1.0, edgecolor=COLORS[b], linewidth=1.2,
           error_kw={"ecolor":"#333333"})
    top = max(a.mean()+a.std(ddof=1)/np.sqrt(len(a)),
              c.mean()+c.std(ddof=1)/np.sqrt(len(c)))
    ax.plot([x[i]-w/2, x[i]+w/2],[top+.025]*2, color="#333333", lw=1)
    ax.text(x[i], top+.035, s, ha="center", fontsize=13,
            color="#333333", weight="bold" if s!="n.s." else "normal")

ax.set_xticks(x)
ax.set_xticklabels([f"{b}\n{BLOCK_DESC[b]}" for b in ORDER], fontsize=12)
ax.set_xlabel("Feature block", labelpad=10)
ax.set_ylabel("Missing rate (mean ± SEM)")
ax.set_ylim(0, 1.18); ax.grid(alpha=.25, axis="y"); ax.set_axisbelow(True)
for side in ("top","right"): ax.spines[side].set_visible(False)
ax.set_title("Missing rate by feature block — patient vs control\n"
             f"(subject-level; control n={n0:,}, patient n={n1:,}; "
             "Mann-Whitney U, two-sided)", weight="bold", pad=14)

# legend: 블록 색이 아닌 '채움 규칙' 자체를 설명하는 중립 회색 프록시
handles = [Patch(facecolor="#b0b0b0", edgecolor="#555555", hatch="//", alpha=.55,
                 label=f"Control  (n={n0:,})"),
           Patch(facecolor="#555555", edgecolor="#555555",
                 label=f"Patient  (n={n1:,})")]
leg = ax.legend(handles=handles, loc="upper left", framealpha=.95, fontsize=12,
                title="Fill = group   ·   Colour = feature block")
leg.get_title().set_fontsize(11); leg.get_title().set_color("#555555")
plt.tight_layout(); plt.savefig(OUT/"plots/fig4_missing_vs_label.png", dpi=120, bbox_inches="tight")
print(tab.to_string(index=False))
print("\n✓ analysis_72/plots/fig4_missing_vs_label.png")
