import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os

# --- CONFIGURATION ---
INPUT_FOLDER = '/home/farooq/Downloads/Overleaf Projects (1 items)/VLM Robustness (ARR Jan 5th)/freequencyResults/seedBench/sample_47628'   # Name of folder containing your PNGs
OUTPUT_FOLDER = '/home/farooq/Downloads/Overleaf Projects (1 items)/VLM Robustness (ARR Jan 5th)/freequencyResults/seedBench/sample_47628/pdf'     # Where PDFs will be saved
import img2pdf
import os



# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Scanning folder: {INPUT_FOLDER}...")

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Prepare paths
            img_path = os.path.join(INPUT_FOLDER, filename)
            new_filename = os.path.splitext(filename)[0] + ".pdf"
            save_path = os.path.join(OUTPUT_FOLDER, new_filename)
            
            # Convert directly (Lossless)
            with open(save_path, "wb") as f:
                f.write(img2pdf.convert(img_path))
            
            print(f"Saved (Lossless): {new_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Done. Quality matches original files exactly.")