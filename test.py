import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'wandb_export_2026-04-05T19_53_54.485-04_00.csv'
df = pd.read_csv(csv_path)
df = df[df['State'] == 'finished'].copy()

METHOD_COL = 'experiment.name'
METRICS = {
    'CIFAR-10\nAccuracy (%)': 'test/accuracy',
    'CIFAR-10\nECE': 'test/ece',
    'CIFAR-10\nNLL': 'test/nll',
    'CIFAR-10-C\nAccuracy (%)': 'test/corrupted_accuracy',
    'CIFAR-10-C\nECE': 'test/corrupted_ece',
    #'CIFAR-10-C\nNLL': 'test/corrupted_nll',
    'SVHN (OOD)\nAUPR (%)': 'ood/svhn/ds_aupr',
    'CIFAR-100 (OOD)\nAUPR (%)': 'ood/cifar100/ds_aupr',
    'SVHN (OOD)\nmp AUPR (%)' : 'ood/svhn/mp_aupr',
    'CIFAR-100 (OOD)\nmpAUPR (%)': 'ood/cifar100/mp_aupr'
}

for nice_name, col in METRICS.items():
    if col in df.columns:
        df[nice_name] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[nice_name] = float('nan')

grouped = df.groupby(METHOD_COL)[list(METRICS.keys())].agg(['mean', 'std'])

formatted_table = pd.DataFrame(index=grouped.index)

for nice_name in METRICS.keys():
    mean_series = grouped[(nice_name, 'mean')]
    std_series = grouped[(nice_name, 'std')]
    
    if 'Accuracy' in nice_name or 'AUPR' in nice_name:
        mean_vals = mean_series * 100
        std_vals = std_series * 100
    else:
        mean_vals = mean_series
        std_vals = std_series
        
    formatted_table[nice_name] = [
        f"{m:.2f} \u00B1 {s:.2f}" if pd.notnull(m) else "N/A"
        for m, s in zip(mean_vals, std_vals)
    ]

formatted_table.index.name = 'Method'
formatted_table.rename(index={
    'cifar10_sngp': 'SNGP',
    'cifar10_sngp_augmented': 'SNGP Augmented',
    'cifar10_supcon_sngp': 'SupCon SNGP'
}, inplace=True)

fig, ax = plt.subplots(figsize=(16, 2.5))
ax.axis('tight')
ax.axis('off')

# Ensure header cells are tall enough
table = ax.table(cellText=formatted_table.values,
                 colLabels=formatted_table.columns,
                 rowLabels=formatted_table.index,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 3)

# make column headers bold
for (row, col), cell in table.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold')

plt.savefig('cifar_results_table.png', bbox_inches='tight', dpi=300)
print(formatted_table.to_markdown())

