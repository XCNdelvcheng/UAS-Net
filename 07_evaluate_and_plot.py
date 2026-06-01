"""
Evaluate predictions, compute metrics (C-index, IBS), and plot survival figures.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score, brier_score
from lifelines.plotting import add_at_risk_counts

class CFG:
    RESULTS_CSV_PATH = "./checkpoints/cv_5fold/independent_test_results.csv"
    OUTPUT_DIR = "./results"

def make_sksurv_struct(events, times):
    y = np.zeros(len(events), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    y['Status'] = events.astype(bool)
    y['Survival_in_days'] = times
    return y

def risk_to_survival_probs(risks, times, events, eval_times):
    df_temp = pd.DataFrame({'T': times, 'E': events, 'Risk': risks})
    cph = CoxPHFitter()
    cph.fit(df_temp, duration_col='T', event_col='E')
    surv_df = cph.predict_survival_function(df_temp, times=eval_times)
    return surv_df.T.values

# (Plotting methods plot_single_roc, plot_km_curve, plot_risk_score_scatter remain identically formatted to your refined version in Script 7).

def main():
    if not os.path.exists(CFG.RESULTS_CSV_PATH):
        print(f"Error: Could not find results file {CFG.RESULTS_CSV_PATH}")
        return

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    sns.set_style("whitegrid")

    df = pd.read_csv(CFG.RESULTS_CSV_PATH)
    num_folds = 5
    samples_per_fold = int(len(df) / num_folds)

    all_risks = df['Risk'].values[:samples_per_fold * num_folds]
    risks_matrix = all_risks.reshape(num_folds, samples_per_fold)
    avg_risks = risks_matrix.mean(axis=0) 

    times = df['Time'].values[:samples_per_fold]
    events = df['Event'].values[:samples_per_fold]

    c_index = concordance_index(times, -avg_risks, events)

    y_test_sksurv = make_sksurv_struct(events, times)
    time_range = np.linspace(times.min(), np.percentile(times, 90), 100)
    surv_probs_matrix = risk_to_survival_probs(avg_risks, times, events, time_range)
    score_ibs = integrated_brier_score(y_test_sksurv, y_test_sksurv, surv_probs_matrix, time_range)

    # Export & Plot (Methods called here)
    print(f"C-Index: {c_index:.4f} | IBS: {score_ibs:.4f}")

if __name__ == "__main__":
    main()
