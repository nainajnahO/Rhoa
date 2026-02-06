"""
Ensemble Voting Visualizer

Creates comparison plots and analyses for the ensemble voting system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent

# Load ensemble results
weights_df = pd.read_csv(script_dir / 'ensemble_weights.csv')
signals_df = pd.read_csv(script_dir / 'ensemble_signals.csv')
threshold_df = pd.read_csv(script_dir / 'ensemble_threshold_analysis.csv')

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Model Weight Distribution (Top 15)
ax1 = plt.subplot(2, 3, 1)
top_15 = weights_df.head(15).sort_values('weight', ascending=True)
colors = ['#2ecc71' if w > 0.05 else '#3498db' for w in top_15['weight']]
ax1.barh(range(len(top_15)), top_15['weight'], color=colors)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['model'], fontsize=9)
ax1.set_xlabel('Normalized Weight', fontweight='bold')
ax1.set_title('Top 15 Model Voting Weights (Precision²)', fontweight='bold', pad=10)
ax1.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax1.text(row['weight'] + 0.002, i, f"{row['weight']:.3f}", 
             va='center', fontsize=8)

# 2. Precision Distribution
ax2 = plt.subplot(2, 3, 2)
precisions = weights_df.sort_values('precision', ascending=False)['precision']
ax2.hist(precisions, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
ax2.axvline(precisions.median(), color='red', linestyle='--', linewidth=2, 
            label=f'Median: {precisions.median():.3f}')
ax2.axvline(precisions.mean(), color='orange', linestyle='--', linewidth=2,
            label=f'Mean: {precisions.mean():.3f}')
ax2.set_xlabel('Individual Model Precision', fontweight='bold')
ax2.set_ylabel('Count', fontweight='bold')
ax2.set_title('Distribution of Model Precisions', fontweight='bold', pad=10)
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Threshold Analysis
ax3 = plt.subplot(2, 3, 3)
ax3_twin = ax3.twinx()

line1 = ax3.plot(threshold_df['top_percent'], threshold_df['precision'], 
                 'o-', color='#2ecc71', linewidth=2, markersize=8, label='Precision')
line2 = ax3.plot(threshold_df['top_percent'], threshold_df['recall'], 
                 's-', color='#e74c3c', linewidth=2, markersize=8, label='Recall')
line3 = ax3.plot(threshold_df['top_percent'], threshold_df['f1'], 
                 '^-', color='#9b59b6', linewidth=2, markersize=8, label='F1')
line4 = ax3_twin.plot(threshold_df['top_percent'], threshold_df['signals'], 
                      'd--', color='#95a5a6', linewidth=2, markersize=7, label='Signals', alpha=0.7)

ax3.set_xlabel('Top % Threshold', fontweight='bold')
ax3.set_ylabel('Performance Metrics', fontweight='bold')
ax3_twin.set_ylabel('Number of Signals', fontweight='bold', color='#95a5a6')
ax3.set_title('Ensemble Performance vs Threshold', fontweight='bold', pad=10)
ax3.set_ylim(0, 1.05)
ax3.grid(alpha=0.3)

lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='lower left')

# 4. Signal Quality Analysis
ax4 = plt.subplot(2, 3, 4)
correct_signals = signals_df[signals_df['actual'] == 1]
incorrect_signals = signals_df[signals_df['actual'] == 0]

ax4.scatter(correct_signals['weighted_score'], correct_signals['n_voters'], 
           s=100, c='#2ecc71', alpha=0.6, edgecolors='darkgreen', linewidth=1.5,
           label=f'Correct (TP={len(correct_signals)})')
if len(incorrect_signals) > 0:
    ax4.scatter(incorrect_signals['weighted_score'], incorrect_signals['n_voters'], 
               s=100, c='#e74c3c', alpha=0.6, edgecolors='darkred', linewidth=1.5,
               label=f'Incorrect (FP={len(incorrect_signals)})')

ax4.set_xlabel('Weighted Vote Score', fontweight='bold')
ax4.set_ylabel('Number of Voting Models', fontweight='bold')
ax4.set_title('Signal Quality Distribution (Top 10%)', fontweight='bold', pad=10)
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Voting Model Participation
ax5 = plt.subplot(2, 3, 5)
avg_voters = signals_df['n_voters'].mean()
median_voters = signals_df['n_voters'].median()
ax5.hist(signals_df['n_voters'], bins=range(0, int(signals_df['n_voters'].max())+2), 
         color='#3498db', edgecolor='black', alpha=0.7)
ax5.axvline(avg_voters, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {avg_voters:.1f}')
ax5.axvline(median_voters, color='orange', linestyle='--', linewidth=2,
            label=f'Median: {median_voters:.1f}')
ax5.set_xlabel('Number of Models Voting', fontweight='bold')
ax5.set_ylabel('Number of Signals', fontweight='bold')
ax5.set_title('Consensus Strength per Signal', fontweight='bold', pad=10)
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Comparison Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Best thresholds
best_precision_row = threshold_df.loc[threshold_df['precision'].idxmax()]
best_f1_row = threshold_df.loc[threshold_df['f1'].idxmax()]
best_recall_row = threshold_df.loc[threshold_df['recall'].idxmax()]

table_data = [
    ['Metric', 'Value', 'Top %'],
    ['', '', ''],
    ['Best Precision', f"{best_precision_row['precision']:.4f}", f"{best_precision_row['top_percent']}%"],
    ['  TP / FP', f"{best_precision_row['tp']} / {best_precision_row['fp']}", ''],
    ['', '', ''],
    ['Best F1', f"{best_f1_row['f1']:.4f}", f"{best_f1_row['top_percent']}%"],
    ['  Precision', f"{best_f1_row['precision']:.4f}", ''],
    ['  Recall', f"{best_f1_row['recall']:.4f}", ''],
    ['', '', ''],
    ['Best Recall', f"{best_recall_row['recall']:.4f}", f"{best_recall_row['top_percent']}%"],
    ['  TP / FN', f"{best_recall_row['tp']} / {89 - best_recall_row['tp']}", ''],
    ['', '', ''],
    ['Active Models', f"{len(weights_df[weights_df['weight'] > 0])}/29", ''],
    ['Avg Voters/Signal', f"{signals_df['n_voters'].mean():.1f}", ''],
]

table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                 colWidths=[0.45, 0.3, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(3):
        if table_data[i][0] == '':
            table[(i, j)].set_facecolor('#ecf0f1')
        elif table_data[i][0].startswith('  '):
            table[(i, j)].set_facecolor('#f8f9fa')
        else:
            table[(i, j)].set_facecolor('white')
            table[(i, j)].set_text_props(weight='bold')

ax6.set_title('Ensemble Performance Summary', fontweight='bold', pad=20, fontsize=12)

plt.suptitle('Weighted Ensemble Voting Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_path = script_dir / 'ensemble_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Ensemble analysis saved to: {output_path}")
plt.close()

# Generate detailed report
print("\n" + "="*80)
print("ENSEMBLE VOTING SYSTEM - DETAILED REPORT")
print("="*80)

print("\n### MODEL WEIGHTS ###")
print(f"\nTop 10 Most Influential Models:")
for idx, row in weights_df.head(10).iterrows():
    print(f"  {row['model']:25s} Weight={row['weight']:.4f} (Precision={row['precision']:.4f})")

print(f"\n### THRESHOLD ANALYSIS ###")
print(threshold_df.to_string(index=False))

print(f"\n### SIGNAL QUALITY ###")
print(f"Total signals at Top 10%: {len(signals_df)}")
print(f"Correct predictions (TP): {len(correct_signals)} ({len(correct_signals)/len(signals_df)*100:.1f}%)")
print(f"Incorrect predictions (FP): {len(incorrect_signals)} ({len(incorrect_signals)/len(signals_df)*100:.1f}%)")
print(f"Average weighted score: {signals_df['weighted_score'].mean():.4f}")
print(f"Average models voting: {signals_df['n_voters'].mean():.1f}")

print(f"\n### TOP VOTED SIGNALS ###")
top_signals = signals_df.nlargest(10, 'weighted_score')
print(f"{'Date':12s} {'Score':>8s} {'Voters':>8s} {'Actual':>8s} {'Top Voter'}")
print("-" * 70)
for idx, row in top_signals.iterrows():
    result = "✓ TP" if row['actual'] == 1 else "✗ FP"
    print(f"{str(row['date'])[:10]:12s} {row['weighted_score']:8.4f} {row['n_voters']:8} {result:8s} {row['top_voter']}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
