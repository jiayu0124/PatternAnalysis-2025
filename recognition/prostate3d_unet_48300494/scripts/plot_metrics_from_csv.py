import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

metrics_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'metrics.csv')
out_png = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'curves_from_csv.png')

if not os.path.exists(metrics_path):
    print('METRICS_MISSING')
    raise SystemExit(1)

df = pd.read_csv(metrics_path)
if df.empty:
    print('METRICS_EMPTY')
    raise SystemExit(1)

epochs = df['epoch'].values
train_loss = df['train_loss'].astype(float).values
val_loss = df['val_loss'].astype(float).values
val_dice = df['val_dice'].astype(float).values

fig, ax1 = plt.subplots(figsize=(6,4))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.plot(epochs, train_loss, label='Train Loss', marker='o')
ax1.plot(epochs, val_loss, label='Val Loss', marker='o')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Dice')
ax2.plot(epochs, val_dice, label='Val Dice', linestyle='--', marker='o')
ax2.legend(loc='upper right')
plt.title('Training Curves (from metrics.csv)')
plt.tight_layout()
plt.savefig(out_png)
print('SAVED', out_png)

