from load_data import load_from_folder
from extract_poi import run_all

print("="*60)
print("TRAVEL TRENDS DATA PIPELINE")
print("="*60)

# Step 1: Load data from CSVs
print("\n[STEP 1] Loading data from CSV files")
load_from_folder("data")

# Step 2: Extract POIs using LLM
print("\n[STEP 2] Extracting POIs from posts")
run_all(unprocessed_only=True)

print("\n[DONE] Pipeline complete!")