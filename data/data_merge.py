import json
import glob

DATA_DIR = "/root/gpufree-data/lapped-lora/data"
OUTPUT_FILE = f"{DATA_DIR}/subset_global.json"

def merge_datasets():
    all_texts = set() # 使用 set 去重，因为我们之前构造了 Overlap
    files = glob.glob(f"{DATA_DIR}/subset_*.json")
    
    print(f"Merging {len(files)} files...")
    
    for fpath in files:
        if "global" in fpath: continue
        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                all_texts.add(item['text'])
    
    # 转回 list 格式
    merged_data = [{"text": t} for t in all_texts]
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
    print(f"Global dataset created with {len(merged_data)} unique samples.")

if __name__ == "__main__":
    merge_datasets()