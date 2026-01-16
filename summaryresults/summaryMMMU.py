import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# --- 1. Configuration ---
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

# --- 2. Data Parsing Loop ---
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
        # Isolate the section before "Confusion vs ground truth"
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
                    # Look for lines like: "Translation 0.059 ..."
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
            # Alternate parsing if the first table is missing
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
                    # Look for lines like: "Translation 0.059 ..."
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
                # Stop if we hit the next section (usually starts with "======")
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
                            
                            # Calculate percentages
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
        # PART C: Detailed Metrics (Robust Parsing)
        # ====================================================
        # We use a state machine to read line-by-line and accumulate text for each perturbation
        lines = content.split('\n')
        current_section = None
        current_pert = None
        
        # Buffers to hold multi-line text for each metric type
        buffers = {
            'DIRICHLET': {pt: "" for pt in pert_types},
            'FREQUENCY': {pt: "" for pt in pert_types},
            'EMBEDDING': {pt: "" for pt in pert_types}
        }
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean: continue
            
            # Detect Section Headers
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
            if "======" in line: # Reset if hitting other headers
                current_section = None
                current_pert = None
                continue
                
            # Detect Perturbation Label inside a section
            if current_section:
                for pt in pert_types:
                    if line_clean.startswith(pt):
                        # Ensure it's a full word match (e.g. avoid matching "Scale" inside "Scale+Pad")
                        remaining = line_clean[len(pt):]
                        if not remaining or remaining[0] in [' ', '\t', '|', ':']:
                            current_pert = pt
                            break
                
                # Accumulate text if we are inside a perturbation block
                if current_pert:
                    buffers[current_section][current_pert] += " " + line_clean

        # Extract values using regex from the accumulated buffers
        for section, perts_data in buffers.items():
            for pt, text in perts_data.items():
                if not text: continue
                
                if section == 'DIRICHLET':
                    # Pattern: ΔE mean= number
                    m = re.search(r'ΔE mean=\s*([-\d\.]+)', text)
                    if m:
                        detailed_metrics_data.append({
                            'Model': model_name, 'Perturbation': pt, 
                            'MetricType': 'Dirichlet', 'Value': float(m.group(1))
                        })
                elif section == 'FREQUENCY':
                    # Pattern: low= number (supports scientific notation)
                    m = re.search(r'low=\s*([-\d\.eE\+]+)', text)
                    if m:
                        detailed_metrics_data.append({
                            'Model': model_name, 'Perturbation': pt, 
                            'MetricType': 'Frequency', 'Value': float(m.group(1))
                        })
                elif section == 'EMBEDDING':
                    # Pattern: ctx-mcq cos= number
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

# --- 4. Plotting ---
sns.set_style("whitegrid")

# FIGURE 1: Base Accuracy & Flip Rate
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
sns.barplot(data=df_acc, x='Model', y='Base Accuracy', palette='viridis')
plt.title('Base Accuracy vs Ground Truth (Higher is Better)')
plt.ylim(0, 0.5) 
for index, row in df_acc.iterrows():
    plt.text(index, row['Base Accuracy'] + 0.01, f"{row['Base Accuracy']:.3f}", 
             color='black', ha="center", fontweight='bold')

plt.subplot(2, 1, 2)
sns.barplot(data=df_rob, x='Perturbation', y='Avg Flip Rate', hue='Model', palette='viridis', 
            order=sorted(df_rob['Perturbation'].unique()))
plt.title('Robustness: Average Flip Rate (Lower is Better)')
plt.legend(title='Model Size')
plt.tight_layout()
plt.savefig('mmmu_fig1_accuracy_robustness.png')
plt.show()

# FIGURE 2: Confusion Matrix Rates
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
p_order = sorted(df_conf['Perturbation'].unique())

# R->W (Error Injection)
sns.barplot(data=df_conf, x='Perturbation', y='R->W Rate (%)', hue='Model', palette='magma', ax=axes[0, 0], order=p_order)
axes[0, 0].set_title('Error Injection Rate (R -> W) (Lower is Better)')
axes[0, 0].set_ylabel('% Originally Correct flipped to Wrong')

# R->R (Stability)
sns.barplot(data=df_conf, x='Perturbation', y='R->R Rate (%)', hue='Model', palette='viridis', ax=axes[0, 1], order=p_order)
axes[0, 1].set_title('Stability of Correct Predictions (R -> R) (Higher is Better)')
axes[0, 1].set_ylim(50, 100)

# W->W (Stubbornness)
sns.barplot(data=df_conf, x='Perturbation', y='W->W Rate (%)', hue='Model', palette='rocket', ax=axes[1, 0], order=p_order)
axes[1, 0].set_title('Persistence of Errors (W -> W)')
axes[1, 0].set_ylim(50, 100)

# W->R (Correction)
sns.barplot(data=df_conf, x='Perturbation', y='W->R Rate (%)', hue='Model', palette='coolwarm', ax=axes[1, 1], order=p_order)
axes[1, 1].set_title('Correction Rate (W -> R)')

plt.tight_layout()
plt.savefig('mmmu_fig2_confusion_rates.png')
plt.show()

# FIGURE 3: Detailed Metrics
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Embedding
df_emb = df_details[df_details['MetricType'] == 'Embedding']
if not df_emb.empty:
    sns.barplot(data=df_emb, x='Perturbation', y='Value', hue='Model', palette='viridis', ax=axes[0], order=sorted(df_emb['Perturbation'].unique()))
    axes[0].set_title('Embedding Stability (Context Cosine Similarity) (Higher is Better)')
    axes[0].set_ylim(0.9, 1.0)
    axes[0].legend(loc='lower right')

# Dirichlet
df_dir = df_details[df_details['MetricType'] == 'Dirichlet']
if not df_dir.empty:
    sns.barplot(data=df_dir, x='Perturbation', y='Value', hue='Model', palette='magma', ax=axes[1], order=sorted(df_dir['Perturbation'].unique()))
    axes[1].set_title('Dirichlet Energy Change (Smoothness)')
    axes[1].set_ylabel('ΔE Mean')

# Frequency
df_freq = df_details[df_details['MetricType'] == 'Frequency']
if not df_freq.empty:
    sns.barplot(data=df_freq, x='Perturbation', y='Value', hue='Model', palette='coolwarm', ax=axes[2], order=sorted(df_freq['Perturbation'].unique()))
    axes[2].set_title('Low Frequency Energy Shift')
    axes[2].set_yscale('symlog')
    axes[2].set_ylabel('Energy Delta (SymLog)')

plt.tight_layout()
plt.savefig('mmmu_fig3_detailed_metrics.png')
plt.show()