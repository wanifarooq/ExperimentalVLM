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
#     '2B': '/home/farooq/Downloads/Vlmresults/MMMu/qwen/summaries/mmmu_validation_qwen2B.txt',
#     '4B': '/home/farooq/Downloads/Vlmresults/MMMu/qwen/summaries/mmmu_validation_qwen4B.txt',
#     '8B': '/home/farooq/Downloads/Vlmresults/MMMu/qwen/summaries/mmmu_validation_qwen8B.txt',
#     '32B':'/home/farooq/Downloads/Vlmresults/MMMu/qwen/summaries/mmmu_validation_qwen32B.txt'
# }

files = {
    '0p5B': '/home/farooq/Downloads/Vlmresults/MMMu/llava/summaries/mmmu_validation_0p5B.txt',
    '7B': '/home/farooq/Downloads/Vlmresults/MMMu/llava/summaries/mmmu_validation_7B.txt',
}

# List of perturbation types to track
pert_types = [
    'Scale+Pad', 'Translation', 'Pad/Crop', 'Scale', 
    'TextOverlay', 'BoxOverlay', 'RandomText', 'Rotation', 'Any'
]

# Storage for extracted data
base_accuracy_data = []
robustness_data = []
confusion_data = []
detailed_metrics_data = []

# --- 2. Data Parsing Loop (Unchanged) ---
for model_name, filename in files.items():
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        continue
        
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        # ====================================================
        # PART A: Base Accuracy & Basic Robustness (Flip Rate)
        # ====================================================
        
        # 1. Base Accuracy
        acc_match = re.search(r'Base accuracy vs ground truth:\s+([0-9.]+)', content)
        if acc_match:
            base_accuracy_data.append({
                'Model': model_name, 
                'Base Accuracy': float(acc_match.group(1))
            })
            
        # 2. Avg Flip Rate (First Table)
        if 'Confusion vs ground truth' in content:
            first_section = content.split('Confusion vs ground truth')[0]
            lines = first_section.split('\n')
            in_rob_table = False
            for line in lines:
                if '-----' in line:
                    in_rob_table = True
                    continue
                if not line.strip(): continue
                
                if in_rob_table:
                    parts = line.strip().split()
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
        elif '(#img_affected / #img_used, #changed / #total)' in content:
            table_section = content.split('(#img_affected / #img_used, #changed / #total)')[1]
            lines = table_section.split('\n')
            in_rob_table = False
            for line in lines:
                if '-----' in line:
                    in_rob_table = True
                    continue
                if not line.strip(): continue
                
                if in_rob_table:
                    parts = line.strip().split()
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

        # ====================================================
        # PART B: Confusion Matrix (R->W, W->R)
        # ====================================================
        if "Confusion vs ground truth" in content:
            table_section = content.split("Confusion vs ground truth")[1]
            lines = table_section.split('\n')
            in_conf_table = False
            
            for line in lines:
                if '-----' in line:
                    in_conf_table = True
                    continue
                if in_conf_table and "======" in line:
                    break
                
                if in_conf_table and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            p_type = parts[0]
                            r_w = float(parts[1])
                            w_r = float(parts[2])
                            r_r = float(parts[3])
                            w_w = float(parts[4])
                            
                            total_correct_base = r_w + r_r
                            total_wrong_base = w_r + w_w
                            
                            rw_rate = (r_w / total_correct_base * 100) if total_correct_base > 0 else 0
                            rr_rate = (r_r / total_correct_base * 100) if total_correct_base > 0 else 0
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

        # ====================================================
        # PART C: Detailed Metrics
        # ====================================================
        lines = content.split('\n')
        current_section = None
        current_pert = None
        
        buffers = {
            'DIRICHLET': {pt: "" for pt in pert_types},
            'FREQUENCY': {pt: "" for pt in pert_types},
            'EMBEDDING': {pt: "" for pt in pert_types}
        }
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean: continue
            
            if "DIRICHLET ANALYSIS" in line:
                current_section = 'DIRICHLET'
                current_pert = None
                continue
            if "FREQUENCY ANALYSIS" in line:
                current_section = 'FREQUENCY'
                current_pert = None
                continue
            if "EMBEDDING INVARIANCE" in line:
                current_section = 'EMBEDDING'
                current_pert = None
                continue
            if "======" in line: 
                current_section = None
                current_pert = None
                continue
                
            if current_section:
                for pt in pert_types:
                    if line_clean.startswith(pt):
                        remaining = line_clean[len(pt):]
                        if not remaining or remaining[0] in [' ', '\t', '|', ':']:
                            current_pert = pt
                            break
                if current_pert:
                    buffers[current_section][current_pert] += " " + line_clean

        for section, perts_data in buffers.items():
            for pt, text in perts_data.items():
                if not text: continue
                
                if section == 'DIRICHLET':
                    m = re.search(r'ΔE mean=\s*([-\d\.]+)', text)
                    if m:
                        detailed_metrics_data.append({
                            'Model': model_name, 'Perturbation': pt, 
                            'MetricType': 'Dirichlet', 'Value': float(m.group(1))
                        })
                elif section == 'FREQUENCY':
                    m = re.search(r'low=\s*([-\d\.eE\+]+)', text)
                    if m:
                        detailed_metrics_data.append({
                            'Model': model_name, 'Perturbation': pt, 
                            'MetricType': 'Frequency', 'Value': float(m.group(1))
                        })
                elif section == 'EMBEDDING':
                    m = re.search(r'ctx-mcq cos=\s*([-\d\.]+)', text)
                    if m:
                        detailed_metrics_data.append({
                            'Model': model_name, 'Perturbation': pt, 
                            'MetricType': 'Embedding', 'Value': float(m.group(1))
                        })

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# --- 3. Create DataFrames ---
df_acc = pd.DataFrame(base_accuracy_data)
df_rob = pd.DataFrame(robustness_data)
df_conf = pd.DataFrame(confusion_data)
df_details = pd.DataFrame(detailed_metrics_data)

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
plt.ylim(0, 0.5) 
for index, row in df_acc.iterrows():
    plt.text(index, row['Base Accuracy'] + 0.01, f"{row['Base Accuracy']:.3f}", 
             color='black', ha="center")
save_plot('mmmu_base_accuracy.pdf')

# 2. Avg Flip Rate
plt.figure(figsize=(8, 5)) # Slightly wider for x-labels
sns.barplot(data=df_rob, x='Perturbation', y='Avg Flip Rate', hue='Model', palette='viridis', 
            order=sorted(df_rob['Perturbation'].unique()))
plt.title('Robustness: Average Flip Rate')
plt.xticks(rotation=45)
save_plot('mmmu_flip_rate.pdf')

# 3. Confusion Matrix: R->W
p_order = sorted(df_conf['Perturbation'].unique())
plt.figure(figsize=(8, 5))
sns.barplot(data=df_conf, x='Perturbation', y='R->W Rate (%)', hue='Model', palette='magma', order=p_order)
plt.title('Error Injection Rate (R -> W)')
plt.ylabel('% Originally Correct flipped to Wrong')
plt.xticks(rotation=45)
save_plot('mmmu_confusion_RW.pdf')

# 4. Confusion Matrix: R->R
plt.figure(figsize=(8, 5))
sns.barplot(data=df_conf, x='Perturbation', y='R->R Rate (%)', hue='Model', palette='viridis', order=p_order)
plt.title('Stability of Correct Predictions (R -> R)')
plt.ylim(50, 100)
plt.xticks(rotation=45)
save_plot('mmmu_confusion_RR.pdf')

# 5. Confusion Matrix: W->W
plt.figure(figsize=(8, 5))
sns.barplot(data=df_conf, x='Perturbation', y='W->W Rate (%)', hue='Model', palette='rocket', order=p_order)
plt.title('Persistence of Errors (W -> W)')
plt.ylim(50, 100)
plt.xticks(rotation=45)
save_plot('mmmu_confusion_WW.pdf')

# 6. Confusion Matrix: W->R
plt.figure(figsize=(8, 5))
sns.barplot(data=df_conf, x='Perturbation', y='W->R Rate (%)', hue='Model', palette='coolwarm', order=p_order)
plt.title('Correction Rate (W -> R)')
plt.xticks(rotation=45)
save_plot('mmmu_confusion_WR.pdf')

# 7. Embedding Stability
df_emb = df_details[df_details['MetricType'] == 'Embedding']
if not df_emb.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_emb, x='Perturbation', y='Value', hue='Model', palette='viridis', order=sorted(df_emb['Perturbation'].unique()))
    plt.title('Embedding Stability (Context Cosine)')
    plt.ylim(0.9, 1.0)
    plt.xticks(rotation=45)
    save_plot('mmmu_metric_embedding.pdf')

# 8. Dirichlet
df_dir = df_details[df_details['MetricType'] == 'Dirichlet']
if not df_dir.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_dir, x='Perturbation', y='Value', hue='Model', palette='magma', order=sorted(df_dir['Perturbation'].unique()))
    plt.title('Dirichlet Energy Change')
    plt.ylabel(r'$\Delta$E Mean')
    plt.xticks(rotation=45)
    save_plot('mmmu_metric_dirichlet.pdf')

# 9. Frequency
df_freq = df_details[df_details['MetricType'] == 'Frequency']
if not df_freq.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_freq, x='Perturbation', y='Value', hue='Model', palette='coolwarm', order=sorted(df_freq['Perturbation'].unique()))
    plt.title('Low Frequency Energy Shift')
    plt.yscale('symlog')
    plt.ylabel('Energy Delta (SymLog)')
    plt.xticks(rotation=45)
    save_plot('mmmu_metric_frequency.pdf')

print("All plots saved as separate PDF files.")