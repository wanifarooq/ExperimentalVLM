import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
# ==========================================
# 1. SETUP AND FILE DEFINITIONS
# ==========================================
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

# ==========================================
# 2. ROBUST DATA PARSING
# ==========================================
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

# ==========================================
# 3. CREATE DATAFRAMES
# ==========================================
df_acc = pd.DataFrame(base_acc_data)
df_flip = pd.DataFrame(flip_rate_data)
df_conf = pd.DataFrame(conf_data)
df_detail = pd.DataFrame(detailed_data)

# ==========================================
# 4. PLOTTING
# ==========================================
sns.set_style("whitegrid")

# --- FIG 1: Accuracy & Flip Rate ---
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
if not df_acc.empty:
    sns.barplot(data=df_acc, x='Model', y='Base Accuracy', palette='viridis')
    plt.title('POPE Base Accuracy (Higher is Better)')
    plt.ylim(0.8, 0.9)
    for i, row in df_acc.iterrows():
        plt.text(i, row['Base Accuracy'], f"{row['Base Accuracy']:.3f}", color='black', ha="center", va='bottom')
    

plt.subplot(2, 1, 2)
if not df_flip.empty:
    sns.barplot(data=df_flip, x='Perturbation', y='Avg Flip Rate', hue='Model', palette='viridis',
                order=sorted(df_flip['Perturbation'].unique()))
    plt.title('Average Flip Rate (Lower is Better)')
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left')

plt.tight_layout()
plt.savefig('pope_1_acc_flip.png')
plt.show()

# --- FIG 2: Confusion Matrices (4 Subplots) ---
if not df_conf.empty:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    p_order = sorted(df_conf['Perturbation'].unique())
    
    # 1. TP Stability
    d = df_conf[df_conf['Condition'] == 'TP_Stability']
    sns.barplot(data=d, x='Perturbation', y='R->W Rate (%)', hue='Model', palette='magma', ax=axes[0,0], order=p_order)
    axes[0,0].set_title('TP Stability: True YES flipping to NO (Lower is Better)')
    axes[0,0].set_ylabel('Flip Rate (%)')

    # 2. TN Stability
    d = df_conf[df_conf['Condition'] == 'TN_Stability']
    sns.barplot(data=d, x='Perturbation', y='R->W Rate (%)', hue='Model', palette='magma', ax=axes[0,1], order=p_order)
    axes[0,1].set_title('TN Stability: True NO flipping to YES (Lower is Better)')
    axes[0,1].set_ylabel('Flip Rate (%)')

    # 3. FP Correction
    d = df_conf[df_conf['Condition'] == 'FP_Correction']
    sns.barplot(data=d, x='Perturbation', y='W->R Rate (%)', hue='Model', palette='coolwarm', ax=axes[1,0], order=p_order)
    axes[1,0].set_title('FP Correction: Hallucination YES fixed to NO')
    axes[1,0].set_ylabel('Correction Rate (%)')

    # 4. FN Correction
    d = df_conf[df_conf['Condition'] == 'FN_Correction']
    sns.barplot(data=d, x='Perturbation', y='W->R Rate (%)', hue='Model', palette='coolwarm', ax=axes[1,1], order=p_order)
    axes[1,1].set_title('FN Correction: Missed NO fixed to YES')
    axes[1,1].set_ylabel('Correction Rate (%)')

    plt.tight_layout()
    plt.savefig('pope_2_confusion.png')
    plt.show()

# --- FIG 3: Detailed Metrics ---
if not df_detail.empty:
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    p_order = sorted(df_detail['Perturbation'].unique())
    
    # Drift
    d = df_detail[df_detail['MetricType'] == 'Embedding Drift (ctx_mcq)']
    if not d.empty:
        sns.barplot(data=d, x='Perturbation', y='Value', hue='Model', palette='viridis', ax=axes[0], order=p_order)
        axes[0].set_title('Embedding Drift (ctx_mcq) [Lower is Better]')
        axes[0].set_ylabel('Cosine Distance')
    
    # Dirichlet
    d = df_detail[df_detail['MetricType'] == 'Dirichlet Energy (ΔE)']
    if not d.empty:
        sns.barplot(data=d, x='Perturbation', y='Value', hue='Model', palette='magma', ax=axes[1], order=p_order)
        axes[1].set_title('Dirichlet Energy Change (ΔE)')
        axes[1].set_ylabel('ΔE Mean')
        
    # Frequency
    d = df_detail[df_detail['MetricType'] == 'Frequency (Low Band)']
    if not d.empty:
        sns.barplot(data=d, x='Perturbation', y='Value', hue='Model', palette='coolwarm', ax=axes[2], order=p_order)
        axes[2].set_title('Low Frequency Energy Shift')
        axes[2].set_ylabel('Energy Delta (SymLog)')
        axes[2].set_yscale('symlog')
        
    plt.tight_layout()
    plt.savefig('pope_3_details.png')
    plt.show()