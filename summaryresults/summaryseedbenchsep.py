import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# --- 1. Configuration & ACL Style Setup ---

# ACL Format Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # Standard ACL font
    "font.size": 11,                    # Standard text size
    "axes.titlesize": 12,               # Title size
    "axes.labelsize": 11,               # Axis label size
    "xtick.labelsize": 10,              # Tick label size
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),           # Standard column width figure size
    "pdf.fonttype": 42                  # Ensures fonts are embedded (editable)
})

# Dictionary mapping model labels to filenames
# files = {
#     '2B': '/home/farooq/Downloads/Vlmresults/seedbench/qwen/summaries/seedbench_qwen2B.txt',
#     '4B': '/home/farooq/Downloads/Vlmresults/seedbench/qwen/summaries/seedbench_qwen4B.txt',
#     '8B': '/home/farooq/Downloads/Vlmresults/seedbench/qwen/summaries/seedbench_qwen8B.txt',
#     '32B': '/home/farooq/Downloads/Vlmresults/seedbench/qwen/summaries/seedbench_qwen32B.txt'
# }

files = {
    '0.5B': '/home/farooq/Downloads/Vlmresults/seedbench/llava/summaries/seedbench_0p5B.txt',
    '7B': '/home/farooq/Downloads/Vlmresults/seedbench/llava/summaries/seedbench_7B.txt'
}

# Lists to store extracted data
base_accuracy_data = []
robustness_data = []
confusion_data = []

# --- 2. Data Parsing Loop ---
for model_name, filename in files.items():
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        continue
        
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # --- A. Extract Base Accuracy ---
        # Pattern: "Base accuracy vs ground truth: 0.654"
        acc_match = re.search(r'Base accuracy vs ground truth:\s+([0-9.]+)', content)
        if acc_match:
            acc = float(acc_match.group(1))
            base_accuracy_data.append({'Model': model_name, 'Base Accuracy': acc})
            
        # --- B. Extract Flip Rate (Avg) ---
        # Look for the first table between "---" and "Confusion"
        if 'Confusion vs ground truth' in content:
            first_section = content.split('Confusion vs ground truth')[0]
            lines = first_section.split('\n')
            in_rob_table = False
            
            for line in lines:
                if '-----' in line:
                    in_rob_table = True
                    continue
                if not line.strip():
                    continue
                    
                if in_rob_table:
                    parts = line.strip().split()
                    # valid lines look like: "Translation 0.059 ..."
                    if len(parts) >= 2 and not parts[0].isdigit():
                        try:
                            p_type = parts[0]
                            avg_val = float(parts[1])
                            robustness_data.append({
                                'Model': model_name,
                                'Perturbation': p_type,
                                'Avg Flip Rate': avg_val
                            })
                        except ValueError:
                            continue

        # --- C. Extract Confusion Matrix (R->W, W->R, etc) ---
        if "Confusion vs ground truth" in content:
            table_section = content.split("Confusion vs ground truth")[1]
            lines = table_section.split('\n')
            in_conf_table = False
            
            for line in lines:
                if '-----' in line:
                    in_conf_table = True
                    continue
                if in_conf_table and (not line.strip() or "completed" in line):
                    break
                
                if in_conf_table:
                    # Format: Type R->W W->R R->R W->W GT
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            p_type = parts[0]
                            r_w = float(parts[1])
                            w_r = float(parts[2])
                            r_r = float(parts[3])
                            w_w = float(parts[4])
                            
                            # Calculate percentages
                            # R->W Rate: (R->W) / (R->W + R->R)
                            total_correct_base = r_w + r_r
                            rw_rate = (r_w / total_correct_base * 100) if total_correct_base > 0 else 0
                            rr_rate = (r_r / total_correct_base * 100) if total_correct_base > 0 else 0
                            
                            # W->R Rate: (W->R) / (W->R + W->W)
                            total_wrong_base = w_r + w_w
                            wr_rate = (w_r / total_wrong_base * 100) if total_wrong_base > 0 else 0
                            ww_rate = (w_w / total_wrong_base * 100) if total_wrong_base > 0 else 0
                            
                            confusion_data.append({
                                'Model': model_name,
                                'Perturbation': p_type,
                                'R->W Rate (%)': rw_rate,
                                'R->R Rate (%)': rr_rate,
                                'W->R Rate (%)': wr_rate,
                                'W->W Rate (%)': ww_rate
                            })
                        except ValueError:
                            continue

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# --- 3. Create DataFrames ---
df_acc = pd.DataFrame(base_accuracy_data)
df_rob = pd.DataFrame(robustness_data)
df_conf = pd.DataFrame(confusion_data)

# --- 4. Plotting (Separated & PDF) ---
sns.set_style("whitegrid")

# Helper to save plots cleanly
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# 1. Base Accuracy
plt.figure()
sns.barplot(data=df_acc, x='Model', y='Base Accuracy', palette='viridis')
plt.title('Base Accuracy vs Ground Truth')
plt.ylim(0, 0.8)  
for index, row in df_acc.iterrows():
    plt.text(index, row['Base Accuracy'] + 0.01, f"{row['Base Accuracy']:.3f}", 
             color='black', ha="center")
save_plot('seedbench_base_accuracy.pdf')

# 2. Avg Flip Rate
plt.figure(figsize=(8, 5)) 
sns.barplot(data=df_rob, x='Perturbation', y='Avg Flip Rate', hue='Model', 
            palette='viridis', order=sorted(list(set(df_rob['Perturbation'].unique()))))
plt.title('Robustness: Average Flip Rate')
plt.ylabel('Flip Rate')
plt.xticks(rotation=45)
save_plot('seedbench_flip_rate.pdf')

# 3. Confusion Matrix: R->W
perturbation_order = sorted(list(set(df_conf['Perturbation'].unique())))
plt.figure(figsize=(8, 5))
sns.barplot(data=df_conf, x='Perturbation', y='R->W Rate (%)', hue='Model', 
            palette='magma', order=perturbation_order)
plt.title('Error Injection Rate (R -> W)')
plt.ylabel('% Originally Correct flipped to Wrong')
plt.xticks(rotation=45)
save_plot('seedbench_confusion_RW.pdf')

# 4. Confusion Matrix: R->R
plt.figure(figsize=(8, 5))
sns.barplot(data=df_conf, x='Perturbation', y='R->R Rate (%)', hue='Model', 
            palette='viridis', order=perturbation_order)
plt.title('Stability of Correct Predictions (R -> R)')
plt.ylabel('% Originally Correct kept Correct')
plt.ylim(50, 100) # Zoomed based on your previous code preference
plt.xticks(rotation=45)
save_plot('seedbench_confusion_RR.pdf')

# 5. Confusion Matrix: W->W
plt.figure(figsize=(8, 5))
sns.barplot(data=df_conf, x='Perturbation', y='W->W Rate (%)', hue='Model', 
            palette='rocket', order=perturbation_order)
plt.title('Persistence of Errors (W -> W)')
plt.ylabel('% Originally Wrong kept Wrong')
plt.ylim(50, 100)
plt.xticks(rotation=45)
save_plot('seedbench_confusion_WW.pdf')

# 6. Confusion Matrix: W->R
plt.figure(figsize=(8, 5))
sns.barplot(data=df_conf, x='Perturbation', y='W->R Rate (%)', hue='Model', 
            palette='coolwarm', order=perturbation_order)
plt.title('Correction Rate (W -> R)')
plt.ylabel('% Originally Wrong flipped to Correct')
plt.xticks(rotation=45)
save_plot('seedbench_confusion_WR.pdf')

print("All SeedBench plots saved as separate PDF files.")