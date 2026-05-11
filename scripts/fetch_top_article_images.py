# scripts/fetch_top_article_images.py
import polars as pl
import json
import sys
import zipfile
from pathlib import Path

# --- 1. Verify Kaggle API & Authentication ---
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("CRITICAL ERROR: 'kaggle' library not found.")
    print("Please run: pip install kaggle")
    sys.exit(1)

print("Authenticating with Kaggle API...")
try:
    api = KaggleApi()
    api.authenticate()
    print("Authentication successful.")
except Exception as e:
    print("\nCRITICAL ERROR: Kaggle Authentication Failed.")
    print("Ensure your kaggle.json file is located in your ~/.kaggle/ directory.")
    print(f"System Error: {e}")
    sys.exit(1)

# --- 2. Configuration ---
DATA_DIR  = Path("data/processed")
IMG_DIR   = Path("data/article_images")
IMG_DIR.mkdir(parents=True, exist_ok=True)
TOP_N     = 250   
COMPETITION = "h-and-m-personalized-fashion-recommendations"

# --- 3. Find Top-N Most Recommended Articles ---
print(f"\nFinding top {TOP_N} most recommended articles across all models...")
try:
    lgbm = pl.read_parquet(DATA_DIR / "lgbm_predictions.parquet").select("article_id")
    als  = pl.read_parquet(DATA_DIR / "ALS_predictions.parquet").with_columns(pl.col("article_id").cast(pl.Int32)).select("article_id")
    mba  = pl.read_parquet(DATA_DIR / "MBA_predictions.parquet").select("article_id")
    
    top_int_ids = (
        pl.concat([lgbm, als, mba])
        .group_by("article_id").len()
        .sort("len", descending=True)
        .head(TOP_N)
        ["article_id"].to_list()
    )
    print(f"  Successfully aggregated {len(top_int_ids)} unique Top-N articles.")
except Exception as e:
    print(f"Error loading prediction parquets: {e}")
    sys.exit(1)

# --- 4. Reverse-Map IDs ---
try:
    with open(DATA_DIR / "id_mapping.json") as f:
        mapping = json.load(f)
    reverse_art = {int(v): str(k) for k, v in mapping["articles"].items()}
    
    orig_ids = [reverse_art.get(int_id) for int_id in top_int_ids if reverse_art.get(int_id)]
    print(f"  Successfully mapped to {len(orig_ids)} original 9-digit article IDs.")
except Exception as e:
    print(f"Error loading id_mapping.json: {e}")
    sys.exit(1)

# --- 5. Download Images via Native API ---
print(f"\nInitiating native Kaggle API download to {IMG_DIR.resolve()}...")
downloaded, skipped, failed = 0, 0, 0

for orig_id in orig_ids:
    # BUG FIX: Restore the leading zero to make it 10 digits BEFORE slicing
    padded_id = str(orig_id).zfill(10)          # e.g., "866731001" -> "0866731001"
    
    folder    = padded_id[:3]                   # e.g., "086"
    filename  = f"{padded_id}.jpg"              # e.g., "0866731001.jpg"
    kaggle_path = f"images/{folder}/{filename}"
    out_file  = IMG_DIR / filename

    if out_file.exists():
        skipped += 1
        continue

    try:
        api.competition_download_file(
            competition=COMPETITION,
            file_name=kaggle_path,
            path=str(IMG_DIR),
            force=True,
            quiet=True 
        )
        
        # Kaggle API zips individual files; extract it
        zip_file = IMG_DIR / (filename + ".zip")
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(IMG_DIR)
            zip_file.unlink() 
            
        downloaded += 1
        
        if downloaded % 25 == 0:
            print(f"  Progress: {downloaded} downloaded, {skipped} skipped, {failed} failed...")
            
    except Exception as e:
        failed += 1
        if failed <= 5: 
            print(f"  [!] FAILED to download {padded_id}: {e}")

print("\n" + "="*50)
print(f"DOWNLOAD COMPLETE")
print(f"Success : {downloaded}")
print(f"Skipped : {skipped} (Already existed)")
print(f"Failed  : {failed}")
print("="*50)