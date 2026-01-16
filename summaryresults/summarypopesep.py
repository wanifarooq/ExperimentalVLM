import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# --- 1. Configuration & ACL Style Setup ---

# ACL Format Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),
    "pdf.fonttype": 42
})

# Dictionary mapping model labels to filenames
files = {
    '2B': '/home/farooq/Downloads/Vlmresults/pope/qwen/summaries/summary_pope_adversarial_2B.txt',
    '8B': '/home/farooq/Downloads/Vlmresults/pope/qwen/summaries/summary_pope_adversarial_8B.txt'
}

# Headers for the 4 specific binary confusion tables
conf_headers = {
    'TP_Stability': '[base pred=Yes, gt=Yes]',
    'TN_Stability': '[base pred=No, gt=No]',
    'FP_Correction': '[base pred=Yes, gt=No]',
    'FN_Correction': '[base pred=No, gt=Yes]'
}

pert_types = [
    'Scale+Pad', 'Translation', 'Pad/Crop', 'Scale', 
    'TextOverlay', 'BoxOverlay', 'RandomText', 'Rotation', 'Any'
]

# Data containers
base_acc_data = []
flip_rate_data = []
conf_data = []
detailed_data = []

# --- 2. Robust Data Parsing ---
for model_name, filename in files.items():
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        continue
        
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        current_section = None
        current_conf_type = None
        current_drift_channel = None
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped: continue
            
            # --- A. BASE ACCURACY ---
            if "Base accuracy vs ground truth:" in line:
                m = re.search(r':\s+([0-9.]+)', line)
                if m:
                    base_acc_data.append({'Model': model_name, 'Base Accuracy': float(m.group(1))})
                continue

            # --- B. FLIP RATE (SUMMARY TABLE) ---
            if "Type" in line and "AVg" in line and "Ve" in line:
                current_section = 'FLIP_RATE'
                continue
            
            # --- C. DETECT CONFUSION TABLES ---
            if "Confusion vs ground truth" in line:
                found_specific = False
                for key, substring in conf_headers.items():
                    if substring in line:
                        current_section = 'CONFUSION'
                        current_conf_type = key
                        found_specific = True
                        break
                if not found_specific:
                    current_section = None 
                continue

            # --- D. DETECT DETAILED METRICS SECTIONS ---
            if "FREQUENCY ANALYSIS" in line:
                current_section = 'FREQUENCY'
                continue
            if "DIRICHLET ANALYSIS" in line:
                current_section = 'DIRICHLET'
                continue
            if "EMBEDDING DRIFT ANALYSIS" in line:
                current_section = 'DRIFT'
                current_drift_channel = None
                continue
            
            # Reset section on delimiters
            if "======" in line:
                current_section = None
                continue

            # =========================================
            # PARSING LINES BASED ON CURRENT SECTION
            # =========================================
            
            # 1. Parse Flip Rate Table
            if current_section == 'FLIP_RATE':
                parts = line_stripped.split()
                if len(parts) >= 2 and parts[0] in pert_types:
                    try:
                        flip_rate_data.append({
                            'Model': model_name,
                            'Perturbation': parts[0],
                            'Avg Flip Rate': float(parts[1])
                        })
                    except ValueError: continue

            # 2. Parse Specific Confusion Tables
            elif current_section == 'CONFUSION' and current_conf_type:
                parts = line_stripped.split()
                if len(parts) >= 5 and parts[0] in pert_types:
                    try:
                        r_w = float(parts[1])
                        w_r = float(parts[2])
                        r_r = float(parts[3])
                        w_w = float(parts[4])
                        
                        # Calculate Rates
                        total_correct_base = r_w + r_r
                        total_wrong_base = w_r + w_w
                        
                        rw_rate = (r_w / total_correct_base * 100) if total_correct_base > 0 else 0
                        wr_rate = (w_r / total_wrong_base * 100) if total_wrong_base > 0 else 0
                        
                        conf_data.append({
                            'Model': model_name,
                            'Perturbation': parts[0],
                            'Condition': current_conf_type,
                            'R->W Rate (%)': rw_rate,
                            'W->R Rate (%)': wr_rate
                        })
                    except ValueError: continue

            # 3. Parse Frequency
            elif current_section == 'FREQUENCY':
                for pt in pert_types:
                    if line_stripped.startswith(pt):
                        after_pt = line_stripped[len(pt):]
                        if not after_pt or after_pt[0].isspace():
                            m = re.search(r'low=\s*([-\d\.eE\+]+)', line)
                            if m:
                                detailed_data.append({
                                    'Model': model_name, 'Perturbation': pt,
                                    'MetricType': 'Frequency (Low Band)', 'Value': float(m.group(1))
                                })
                            break 

            # 4. Parse Dirichlet
            elif current_section == 'DIRICHLET':
                for pt in pert_types:
                    if line_stripped.startswith(pt):
                        after_pt = line_stripped[len(pt):]
                        if not after_pt or after_pt[0].isspace():
                            # Parsing fix for "ΔE mean= 7.900±..."
                            m = re.search(r'ΔE mean=\s*([-\d\.eE]+)', line)
                            if m:
                                try:
                                    detailed_data.append({
                                        'Model': model_name, 'Perturbation': pt,
                                        'MetricType': 'Dirichlet Energy (ΔE)', 'Value': float(m.group(1))
                                    })
                                except ValueError: pass
                            break

            # 5. Parse Drift
            elif current_section == 'DRIFT':
                if "Channel:" in line:
                    if "ctx_mcq" in line:
                        current_drift_channel = 'ctx_mcq'
                    else:
                        current_drift_channel = None
                    continue
                
                if current_drift_channel == 'ctx_mcq':
                    parts = line_stripped.split()
                    if len(parts) >= 2 and parts[0] in pert_types:
                        try:
                            detailed_data.append({
                                'Model': model_name, 'Perturbation': parts[0],
                                'MetricType': 'Embedding Drift (ctx_mcq)', 'Value': float(parts[1])
                            })
                        except ValueError: continue

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# --- 3. Create DataFrames ---
df_acc = pd.DataFrame(base_acc_data)
df_flip = pd.DataFrame(flip_rate_data)
df_conf = pd.DataFrame(conf_data)
df_detail = pd.DataFrame(detailed_data)

# --- 4. Plotting (Separated & PDF) ---
sns.set_style("whitegrid")

# Helper to save plots cleanly
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# 1. Base Accuracy
if not df_acc.empty:
    plt.figure()
    sns.barplot(data=df_acc, x='Model', y='Base Accuracy', palette='viridis')
    plt.title('POPE Base Accuracy')
    plt.ylim(0.8, 0.9)
    for i, row in df_acc.iterrows():
        plt.text(i, row['Base Accuracy'], f"{row['Base Accuracy']:.3f}", color='black', ha="center", va='bottom')
    save_plot('pope_base_accuracy.pdf')

# 2. Avg Flip Rate
if not df_flip.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_flip, x='Perturbation', y='Avg Flip Rate', hue='Model', palette='viridis',
                order=sorted(df_flip['Perturbation'].unique()))
    plt.title('Average Flip Rate')
    plt.ylabel('Avg Flip Rate (Lower is Better)')
    plt.xticks(rotation=45)
    save_plot('pope_flip_rate.pdf')

# --- CONFUSION MATRICES (TP/TN Stability, FP/FN Correction) ---
if not df_conf.empty:
    p_order = sorted(df_conf['Perturbation'].unique())
    
    # 3. TP Stability
    d = df_conf[df_conf['Condition'] == 'TP_Stability']
    if not d.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=d, x='Perturbation', y='R->W Rate (%)', hue='Model', palette='magma', order=p_order)
        plt.title('TP Stability: True YES flipping to NO')
        plt.ylabel('Flip Rate (%)')
        plt.xticks(rotation=45)
        save_plot('pope_TP_stability.pdf')

    # 4. TN Stability
    d = df_conf[df_conf['Condition'] == 'TN_Stability']
    if not d.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=d, x='Perturbation', y='R->W Rate (%)', hue='Model', palette='magma', order=p_order)
        plt.title('TN Stability: True NO flipping to YES')
        plt.ylabel('Flip Rate (%)')
        plt.xticks(rotation=45)
        save_plot('pope_TN_stability.pdf')

    # 5. FP Correction
    d = df_conf[df_conf['Condition'] == 'FP_Correction']
    if not d.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=d, x='Perturbation', y='W->R Rate (%)', hue='Model', palette='coolwarm', order=p_order)
        plt.title('FP Correction: Hallucination YES fixed to NO')
        plt.ylabel('Correction Rate (%)')
        plt.xticks(rotation=45)
        save_plot('pope_FP_correction.pdf')

    # 6. FN Correction
    d = df_conf[df_conf['Condition'] == 'FN_Correction']
    if not d.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=d, x='Perturbation', y='W->R Rate (%)', hue='Model', palette='coolwarm', order=p_order)
        plt.title('FN Correction: Missed NO fixed to YES')
        plt.ylabel('Correction Rate (%)')
        plt.xticks(rotation=45)
        save_plot('pope_FN_correction.pdf')

# --- DETAILED METRICS ---
if not df_detail.empty:
    p_order = sorted(df_detail['Perturbation'].unique())
    
    # 7. Embedding Drift
    d = df_detail[df_detail['MetricType'] == 'Embedding Drift (ctx_mcq)']
    if not d.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=d, x='Perturbation', y='Value', hue='Model', palette='viridis', order=p_order)
        plt.title('Embedding Drift (ctx_mcq)')
        plt.ylabel('Cosine Distance')
        plt.xticks(rotation=45)
        save_plot('pope_metric_drift.pdf')
    
    # 8. Dirichlet Energy
    d = df_detail[df_detail['MetricType'] == 'Dirichlet Energy (ΔE)']
    if not d.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=d, x='Perturbation', y='Value', hue='Model', palette='magma', order=p_order)
        plt.title(r'Dirichlet Energy Change ($\Delta$E)')
        plt.ylabel(r'$\Delta$E Mean')
        plt.xticks(rotation=45)
        save_plot('pope_metric_dirichlet.pdf')
        
    # 9. Frequency Analysis
    d = df_detail[df_detail['MetricType'] == 'Frequency (Low Band)']
    if not d.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=d, x='Perturbation', y='Value', hue='Model', palette='coolwarm', order=p_order)
        plt.title('Low Frequency Energy Shift')
        plt.ylabel('Energy Delta (SymLog)')
        plt.yscale('symlog')
        plt.xticks(rotation=45)
        save_plot('pope_metric_frequency.pdf')

print("All POPE plots saved as separate PDF files.")