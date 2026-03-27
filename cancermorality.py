"""
Cancer Morality Project
Developing a random forest model to identify key causes of cancer mortality in the United States.
Integrating CDC data and validating results with cross-validation and analyses to support cancer prevention.
"""
# This import tools. 
import argparse
import warnings
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")  # works without a display — saves PNGs to disk
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
# This code create plot featuring the rate and type of cancers.
RANDOM_STATE = 42
RF_PARAMS = dict(
    n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", bootstrap=True, oob_score=True,
    random_state=RANDOM_STATE, n_jobs=-1,
)
FEATURE_COLS = ["crude_rate", "pct_of_total_deaths", "log_deaths",
                "system_code", "rate_to_deaths_ratio"]
TARGET_COL   = "age_adj_rate"
 
SYSTEM_COLORS = {
    "Respiratory": "#C0392B", "Digestive": "#E67E22", "Breast": "#8E44AD",
    "Male Genital": "#2980B9", "Female Genital": "#E91E8C", "Leukemia": "#16A085",
    "Lymphoma": "#27AE60", "Urinary": "#F39C12", "Brain/Nervous": "#7F8C8D",
    "Miscellaneous": "#95A5A6", "Skin": "#D4AC0D", "Mesothelioma": "#5D6D7E",
    "Oral Cavity & Pharynx": "#A04000", "Endocrine": "#117A65",
    "Bone & Joint": "#884EA0", "Eye": "#138D75", "Myeloma": "#2E4053",
    "Unknown": "#BDC3C7",
}
 
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
 
 
# This code find the path to the CSV file. 
def find_data_file(explicit_path=None):
    """Find the CDC WONDER CSV regardless of working directory."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"File not found:\n  {p}\n"
            "Check the path spelling and try again."
        )
    # Search next to the script, then cwd
    for d in [Path(__file__).parent, Path.cwd()]:
        for name in [
            "United_States_and_Puerto_Rico_Cancer_Statistics_1999-2021_Mortality.csv",
            "United_States_and_Puerto_Rico_Cancer_Statistics__1999-2021_Mortality.csv",
        ]:
            if (d / name).exists():
                return d / name
        # Fuzzy: any CSV with 1999 and 2021 in the name
        for p in d.glob("*.csv"):
            if "1999" in p.name and "2021" in p.name:
                return p
 
    raise FileNotFoundError(
        "\nCould not find the CDC WONDER CSV.\n\n"
        "QUICK FIX — run with --data flag:\n"
        "  python cancer_mortality_rf.py --data \"/Users/marti/Desktop/Cancer Morality/"
        "United_States_and_Puerto_Rico_Cancer_Statistics_1999-2021_Mortality.csv\""
    )
 
 
# This function load and clean the dataset.
def load_cdc_wonder(path):
    df = pd.read_csv(path)
    df.columns = ["notes","cancer_site","code","deaths","population","age_adj_rate","crude_rate"]
    df = df[df["cancer_site"].notna() & df["deaths"].notna()].copy()
    for col in ["deaths","population","age_adj_rate","crude_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_aggregate"] = df["code"].astype(str).str.contains("-")
    df["is_total"]     = df["cancer_site"] == "All Cancer Sites Combined"
    return df
 
def extract_leaf_sites(df):
    leaf = df[~df["is_aggregate"] & ~df["is_total"]].copy()
    leaf = leaf[~leaf["cancer_site"].isin(["Male and Female Breast"])]
    return leaf.dropna(subset=["deaths","age_adj_rate"]).reset_index(drop=True)
 
def assign_body_system(code):
    try:
        c = int(str(code).split("-")[0])
        if   20010<=c<=20100: return "Oral Cavity & Pharynx"
        elif 21010<=c<=21130: return "Digestive"
        elif 22010<=c<=22060: return "Respiratory"
        elif c==23000: return "Bone & Joint"
        elif c==24000: return "Soft Tissue"
        elif 25010<=c<=25020: return "Skin"
        elif c==26000: return "Breast"
        elif 27010<=c<=27070: return "Female Genital"
        elif 28010<=c<=28040: return "Male Genital"
        elif 29010<=c<=29040: return "Urinary"
        elif c==30000: return "Eye"
        elif 31010<=c<=31040: return "Brain/Nervous"
        elif 32010<=c<=32020: return "Endocrine"
        elif 33011<=c<=33042: return "Lymphoma"
        elif c==34000: return "Myeloma"
        elif 35011<=c<=35043: return "Leukemia"
        elif c==36010: return "Mesothelioma"
        else: return "Miscellaneous"
    except: return "Unknown"
 
def engineer_features(leaf, total_deaths):
    leaf = leaf.copy()
    leaf["pct_of_total_deaths"]  = leaf["deaths"] / total_deaths * 100
    leaf["log_deaths"]           = np.log1p(leaf["deaths"])
    leaf["rate_to_deaths_ratio"] = leaf["age_adj_rate"] / leaf["deaths"].replace(0, np.nan) * 1e6
    leaf["body_system"]          = leaf["code"].apply(assign_body_system)
    leaf["system_code"]          = pd.Categorical(leaf["body_system"]).codes
    return leaf
 
 
# This function does the random forest training model. 
def train_random_forest(leaf):
    X = leaf[FEATURE_COLS].fillna(0).values
    y = leaf[TARGET_COL].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_scaled, y)
    y_pred    = rf.predict(X_scaled)
    kf        = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=kf, scoring="r2")
    perm      = permutation_importance(rf, X_scaled, y, n_repeats=30, random_state=RANDOM_STATE)
    return {
        "model": rf, "scaler": scaler, "X_scaled": X_scaled, "y": y, "y_pred": y_pred,
        "metrics": {
            "r2_train":   round(r2_score(y, y_pred), 4),
            "oob_r2":     round(rf.oob_score_, 4),
            "rmse":       round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
            "mae":        round(float(mean_absolute_error(y, y_pred)), 4),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std":  round(float(cv_scores.std()), 4),
        },
        "cv_scores": cv_scores, "perm_imp": perm,
    }
 
def build_results_table(leaf, results):
    df = leaf[["cancer_site","body_system","deaths","age_adj_rate","crude_rate","pct_of_total_deaths"]].copy()
    df["predicted_rate"] = results["y_pred"]
    df["residual"]       = df["age_adj_rate"] - df["predicted_rate"]
    df["abs_residual"]   = df["residual"].abs()
    df["priority_score"] = df["age_adj_rate"] * np.log1p(df["deaths"])
    return df.sort_values("age_adj_rate", ascending=False).reset_index(drop=True)
 
 
# This function create the plots. 
def plot_dashboard(leaf, results, out_dir):
    rf=results["model"]; y=results["y"]; y_pred=results["y_pred"]
    cv=results["cv_scores"]; perm=results["perm_imp"]; m=results["metrics"]
    site_names=leaf["cancer_site"].values
    sc_colors=[SYSTEM_COLORS.get(s,"#BDC3C7") for s in leaf["body_system"]]
    BLUE,RED,TEAL,GOLD,GRAY="#1B4F8A","#C0392B","#1A7A6E","#D4A017","#6B7280"
    feat_labels={"crude_rate":"Crude Rate","pct_of_total_deaths":"% of Total Deaths",
                 "log_deaths":"Log(Deaths)","rate_to_deaths_ratio":"Rate/Deaths Ratio","system_code":"Body System"}
 
    fig=plt.figure(figsize=(22,26)); gs=gridspec.GridSpec(4,3,figure=fig,hspace=0.48,wspace=0.38)
 
    # P1: top 15 by deaths
    ax=fig.add_subplot(gs[0,:2])
    top=leaf.nlargest(15,"deaths")[["cancer_site","deaths","body_system"]].sort_values("deaths")
    cols=[SYSTEM_COLORS.get(s,"#BDC3C7") for s in top["body_system"]]
    bars=ax.barh(top["cancer_site"],top["deaths"]/1e6,color=cols,edgecolor="none",height=0.65)
    for b,v in zip(bars,top["deaths"]): ax.text(b.get_width()+0.02,b.get_y()+b.get_height()/2,f"{v/1e6:.2f}M",va="center",fontsize=8.5,color=GRAY)
    ax.set_xlabel("Total Deaths 1999–2021 (millions)",fontsize=11); ax.set_title("Top 15 Cancer Sites by Total Mortality (1999–2021)",fontsize=12,fontweight="bold")
    ax.xaxis.grid(True,alpha=0.25,linestyle="--"); ax.set_axisbelow(True)
 
    # P2: top 12 by rate
    ax=fig.add_subplot(gs[0,2])
    top_r=leaf.nlargest(12,"age_adj_rate")[["cancer_site","age_adj_rate"]].sort_values("age_adj_rate")
    cmap2=plt.cm.RdYlBu_r(np.linspace(0.15,0.85,len(top_r)))
    ax.barh(top_r["cancer_site"],top_r["age_adj_rate"],color=cmap2,edgecolor="none",height=0.65)
    for i,(v,_) in enumerate(zip(top_r["age_adj_rate"],top_r["cancer_site"])): ax.text(v+0.3,i,f"{v:.1f}",va="center",fontsize=8,color=GRAY)
    ax.set_xlabel("Age-Adjusted Rate (per 100k)",fontsize=10); ax.set_title("Mortality Rate by\nCancer Site",fontsize=11,fontweight="bold")
    ax.xaxis.grid(True,alpha=0.25,linestyle="--"); ax.set_axisbelow(True)
 
    # P3: predicted vs actual
    ax=fig.add_subplot(gs[1,0])
    ax.scatter(y,y_pred,c=sc_colors,s=60,alpha=0.8,edgecolors="white",linewidth=0.4,zorder=3)
    lo,hi=min(y.min(),y_pred.min())*0.9,max(y.max(),y_pred.max())*1.05
    ax.plot([lo,hi],[lo,hi],"--",color=GRAY,linewidth=1.5,zorder=2)
    for i in np.argsort(np.abs(y-y_pred))[::-1][:4]:
        ax.annotate(site_names[i].split(" ")[0],xy=(y[i],y_pred[i]),xytext=(4,4),textcoords="offset points",fontsize=7.5)
    ax.set_xlabel("Observed Rate (per 100k)",fontsize=10); ax.set_ylabel("Predicted Rate (per 100k)",fontsize=10)
    ax.set_title(f"Predicted vs. Observed\nR² = {m['r2_train']:.3f}  |  RMSE = {m['rmse']:.2f}",fontsize=11,fontweight="bold")
 
    # P4: MDI
    ax=fig.add_subplot(gs[1,1])
    mdi=rf.feature_importances_; order=np.argsort(mdi)
    ic=plt.cm.Blues(np.linspace(0.35,0.85,len(FEATURE_COLS)))
    bars4=ax.barh([feat_labels[FEATURE_COLS[i]] for i in order],mdi[order],color=ic,edgecolor="none",height=0.6)
    for b,v in zip(bars4,mdi[order]): ax.text(b.get_width()+0.003,b.get_y()+b.get_height()/2,f"{v:.3f}",va="center",fontsize=9)
    ax.set_xlabel("MDI Feature Importance",fontsize=10); ax.set_title("Feature Importance\n(Mean Decrease Impurity)",fontsize=11,fontweight="bold")
    ax.xaxis.grid(True,alpha=0.25,linestyle="--"); ax.set_axisbelow(True)
 
    # P5: permutation
    ax=fig.add_subplot(gs[1,2])
    pm=perm.importances_mean; ps=perm.importances_std; o5=np.argsort(pm)
    pc=plt.cm.Oranges(np.linspace(0.35,0.85,len(FEATURE_COLS)))
    ax.barh([feat_labels[FEATURE_COLS[i]] for i in o5],pm[o5],xerr=ps[o5],color=pc,edgecolor="none",error_kw={"linewidth":1.2,"ecolor":"#666"},height=0.6)
    ax.set_xlabel("Mean R² Decrease",fontsize=10); ax.set_title("Permutation Importance\n(30 repeats)",fontsize=11,fontweight="bold")
    ax.xaxis.grid(True,alpha=0.25,linestyle="--"); ax.set_axisbelow(True)
 
    # P6: 5-fold CV
    ax=fig.add_subplot(gs[2,0])
    folds=np.arange(1,len(cv)+1)
    ax.plot(folds,cv,"o-",color=BLUE,linewidth=2,markersize=8,zorder=3)
    ax.fill_between(folds,cv.mean()-cv.std(),cv.mean()+cv.std(),alpha=0.15,color=BLUE)
    ax.axhline(cv.mean(),color=RED,linewidth=1.5,linestyle="--",label=f"Mean R²={cv.mean():.3f}")
    ax.set_xticks(folds); ax.set_xlabel("Fold",fontsize=10); ax.set_ylabel("R²",fontsize=10)
    ax.set_title(f"5-Fold Cross-Validation\nMean R² = {m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f}",fontsize=11,fontweight="bold")
    ax.legend(fontsize=9); ax.yaxis.grid(True,alpha=0.25,linestyle="--"); ax.set_axisbelow(True)
 
    # P7: system breakdown
    ax=fig.add_subplot(gs[2,1])
    sd=leaf.groupby("body_system")["deaths"].sum().sort_values()
    sc7=[SYSTEM_COLORS.get(s,"#BDC3C7") for s in sd.index]
    bars7=ax.barh(sd.index,sd/1e6,color=sc7,edgecolor="none",height=0.65)
    for b,v in zip(bars7,sd): ax.text(b.get_width()+0.02,b.get_y()+b.get_height()/2,f"{v/1e6:.2f}M",va="center",fontsize=7.5,color=GRAY)
    ax.set_xlabel("Total Deaths (millions)",fontsize=10); ax.set_title("Deaths by\nBody System",fontsize=11,fontweight="bold")
    ax.xaxis.grid(True,alpha=0.25,linestyle="--"); ax.set_axisbelow(True)
 
    # P8: residuals
    ax=fig.add_subplot(gs[2,2])
    res=y-y_pred
    ax.scatter(y_pred,res,c=sc_colors,s=50,alpha=0.75,edgecolors="white",linewidth=0.3)
    ax.axhline(0,color=RED,linewidth=1.5,linestyle="--")
    ax.axhline(res.std(),color=GRAY,linewidth=1,linestyle=":",alpha=0.7)
    ax.axhline(-res.std(),color=GRAY,linewidth=1,linestyle=":",alpha=0.7)
    for i in np.argsort(np.abs(res))[::-1][:3]:
        ax.annotate(site_names[i].split(" ")[0],xy=(y_pred[i],res[i]),xytext=(4,4),textcoords="offset points",fontsize=7.5)
    ax.set_xlabel("Predicted Rate (per 100k)",fontsize=10); ax.set_ylabel("Residual (Observed − Predicted)",fontsize=10)
    ax.set_title("Residual Diagnostics",fontsize=11,fontweight="bold")
 
    # P9: metric cards
    ax=fig.add_subplot(gs[3,:])
    ax.set_xlim(0,10); ax.set_ylim(0,1); ax.axis("off")
    cards=[("Training R²",f"{m['r2_train']}",BLUE),("OOB R²",f"{m['oob_r2']}",TEAL),
           ("5-Fold CV R²",f"{m['cv_r2_mean']} ± {m['cv_r2_std']}",GOLD),
           ("RMSE",f"{m['rmse']} / 100k",RED),("MAE",f"{m['mae']} / 100k","#8E44AD"),("Cancer Sites",f"{len(leaf)}",GRAY)]
    cw=10/len(cards)
    for i,(lbl,val,col) in enumerate(cards):
        x=i*cw+cw/2
        ax.add_patch(mpatches.FancyBboxPatch((i*cw+0.08,0.12),cw-0.16,0.76,boxstyle="round,pad=0.03",facecolor=col+"22",edgecolor=col,linewidth=1.5))
        ax.text(x,0.72,lbl,ha="center",va="center",fontsize=9.5,color=col,fontweight="bold")
        ax.text(x,0.35,val,ha="center",va="center",fontsize=13,color="#1a1a1a",fontweight="bold")
    ax.set_title("Model Performance Summary",fontsize=12,fontweight="bold",pad=8)
 
    fig.suptitle("Random Forest Analysis — U.S. Cancer Mortality (CDC WONDER, 1999–2021)\nIdentifying Key Causes of Cancer Mortality to Support Prevention",fontsize=15,fontweight="bold",y=0.995)
    out=out_dir/"cancer_rf_analysis.png"
    plt.savefig(out,dpi=180,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"  Saved: {out}")
 
 
def plot_prevention_priority(leaf, results, out_dir):
    y=results["y"]; y_pred=results["y_pred"]; site_names=leaf["cancer_site"].values
    sc_colors=[SYSTEM_COLORS.get(s,"#BDC3C7") for s in leaf["body_system"]]
    BLUE,RED="#1B4F8A","#C0392B"
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,9)); fig.patch.set_facecolor("white")
 
    for sys in sorted(set(leaf["body_system"])):
        mask=leaf["body_system"].values==sys
        ax1.scatter(y[mask],y_pred[mask],label=sys,color=SYSTEM_COLORS.get(sys,"#BDC3C7"),s=70,alpha=0.85,edgecolors="white",linewidth=0.5,zorder=3)
    lo,hi=min(y.min(),y_pred.min())*0.85,max(y.max(),y_pred.max())*1.08
    ax1.plot([lo,hi],[lo,hi],"--",color="#555",linewidth=1.5,zorder=2)
    for i,(obs,pred,name) in enumerate(zip(y,y_pred,site_names)):
        if obs>8 or abs(obs-pred)>4:
            ax1.annotate(name.split(" ")[0] if len(name)>20 else name,xy=(obs,pred),xytext=(5,3),textcoords="offset points",fontsize=7.5)
    ax1.set_xlabel("Observed Age-Adjusted Rate (per 100,000)",fontsize=12); ax1.set_ylabel("Predicted Age-Adjusted Rate (per 100,000)",fontsize=12)
    ax1.set_title("RF Predicted vs. Observed Mortality Rates\nColored by Body System",fontsize=12,fontweight="bold")
    ax1.legend(loc="upper left",fontsize=7.5,ncol=2,framealpha=0.85,title="Body System",title_fontsize=8)
    ax1.xaxis.grid(True,alpha=0.2,linestyle="--"); ax1.set_axisbelow(True)
    ax1.text(0.97,0.05,f"R² = {results['metrics']['r2_train']:.4f}",transform=ax1.transAxes,ha="right",fontsize=11,color=BLUE,fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4",facecolor="#E8F0FA",edgecolor=BLUE,linewidth=1.2))
 
    top15=leaf.nlargest(15,"age_adj_rate").sort_values("age_adj_rate",ascending=True).reset_index(drop=True)
    sizes=(top15["deaths"]/top15["deaths"].max()*900+80).values
    colors=[SYSTEM_COLORS.get(s,"#BDC3C7") for s in top15["body_system"]]
    for i,row in enumerate(top15.itertuples()):
        ax2.scatter(row.age_adj_rate,i,s=sizes[i],color=colors[i],alpha=0.8,edgecolors="white",linewidth=0.5,zorder=3)
        ax2.text(row.age_adj_rate+0.5,i,f"{row.age_adj_rate:.1f}  ({row.deaths/1e3:.0f}k deaths)",va="center",fontsize=8,color="#333")
    ax2.set_yticks(range(len(top15))); ax2.set_yticklabels(top15["cancer_site"],fontsize=8.5)
    ax2.set_xlabel("Age-Adjusted Mortality Rate (per 100,000)",fontsize=12)
    ax2.set_title("Cancer Prevention Priority Matrix\nBubble size = total deaths  |  X-axis = mortality rate urgency",fontsize=12,fontweight="bold")
    ax2.xaxis.grid(True,alpha=0.2,linestyle="--"); ax2.set_axisbelow(True)
    ax2.annotate("Lung & Bronchus:\nhighest rate AND highest deaths\n→ #1 prevention priority",
                 xy=(44.8,14),xytext=(22,10),arrowprops=dict(arrowstyle="->",color=RED,lw=1.5),fontsize=8.5,color=RED,ha="center")
    fig.suptitle("Cancer Mortality Analysis — Key Findings for Prevention Planning\nCDC WONDER 1999–2021 | Random Forest Model",fontsize=13,fontweight="bold",y=1.01)
    plt.tight_layout()
    out=out_dir/"cancer_prevention_priority.png"
    plt.savefig(out,dpi=180,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"  Saved: {out}")
 
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser=argparse.ArgumentParser(description="Cancer Mortality Random Forest")
    parser.add_argument("--data",type=str,default=None,help="Path to CDC WONDER CSV")
    args=parser.parse_args()
 
    data_path=find_data_file(args.data)
    print(f"Data file: {data_path}")
    out_dir=Path(__file__).parent/"results"; out_dir.mkdir(exist_ok=True)
 
    print("="*60); print("Cancer Mortality Random Forest Analysis"); print("="*60)
 
    df=load_cdc_wonder(data_path)
    leaf=extract_leaf_sites(df)
    total_deaths=df[df["is_total"]]["deaths"].values[0]
    leaf=engineer_features(leaf,total_deaths)
    print(f"Loaded: {len(leaf)} cancer sites | Total deaths: {total_deaths:,.0f}")
 
    results=train_random_forest(leaf)
    res_table=build_results_table(leaf,results)
    m=results["metrics"]
 
    print(f"\nModel Performance:")
    for k,v in m.items(): print(f"  {k:<15s}: {v}")
 
    print(f"\nFeature Importances (Permutation):")
    perm=results["perm_imp"]
    for feat,mean,std in sorted(zip(FEATURE_COLS,perm.importances_mean,perm.importances_std),key=lambda x:-x[1]):
        print(f"  {feat:<30s}  {mean:.4f} ± {std:.4f}")
 
    print(f"\nTop 10 Cancer Sites by Mortality Rate:")
    print(res_table[["cancer_site","age_adj_rate","predicted_rate","pct_of_total_deaths"]].head(10).to_string(index=False))
 
    print(f"\nTop 5 Prevention Priorities:")
    for _,row in res_table.nlargest(5,"priority_score").iterrows():
        print(f"  {row['cancer_site']:<35s}  Rate={row['age_adj_rate']:.1f}  Deaths={row['deaths']/1e6:.2f}M")
 
    csv_out=out_dir/"predictions.csv"; res_table.to_csv(csv_out,index=False)
    print(f"\nSaved: {csv_out}")
 
    print("\nGenerating figures...")
    plot_dashboard(leaf,results,out_dir)
    plot_prevention_priority(leaf,results,out_dir)
    print(f"\n✓ Done. All outputs in: {out_dir}/")
 
if __name__=="__main__":
    main()