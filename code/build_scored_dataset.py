import torch
import os
import json
import argparse
from datasets import load_dataset
from typing import List, Dict, Tuple, Set, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, AutoModelForSeq2SeqLM
from src.eval import initialize_models, judge
from tqdm import tqdm
from peft import PeftModel

logging.set_verbosity_error()

DEFAULT_DATASET_PATH = "theblackcat102/ADL_Final_25W_part1_with_cost"
DEFAULT_SAFETY_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
DEFAULT_USEFULNESS_MODEL = "theblackcat102/Qwen3-1.7B-Usefulness-Judge"
DEFAULT_CHAT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_REWRITE_MODEL = "google/flan-t5-large"       
DEFAULT_INPUT_SCORED = "data/scored_responses.jsonl"
DEFAULT_OUTPUT_SCORED = "data/scored_responses.jsonl"

def parse_args():
    """解析命令行參數，設定模型和訓練配置。"""
    parser = argparse.ArgumentParser("Score Rewritten Prompts")

    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH,
                        help="原始提示的資料集路徑。")
    parser.add_argument("--guard_model", type=str, default=DEFAULT_SAFETY_MODEL)
    parser.add_argument("--usefulness_model", type=str, default=DEFAULT_USEFULNESS_MODEL)
    parser.add_argument("--chat_model", type=str, default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--rewrite_model", type=str, default=DEFAULT_REWRITE_MODEL) 
    parser.add_argument("--lora_path", type=str, default=None)
    # 新增獨立的輸入/輸出參數，滿足不更動原檔案的要求
    parser.add_argument("--input_scored", type=str, default=DEFAULT_INPUT_SCORED,
                        help="現有的單一評分數據檔案路徑 (此檔案不會被修改)。")
    parser.add_argument("--output_scored", type=str, default=DEFAULT_OUTPUT_SCORED,
                        help="輸出包含舊資料和新生成資料的完整檔案路徑 (此檔案會被創建或覆蓋)。")
                        
    parser.add_argument("--iterations", type=int, default=10,
                        help="每個原始提示生成和評分的次數。")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制處理的原始提示數量，用於調試。")
    return parser.parse_args()


def load_original_dataset(path: str):
    """載入原始提示資料集。"""
    print(f"Loading dataset {path} ...")
    ds = load_dataset(path)
    split = list(ds.keys())[0]
    ds = ds[split]
    if 'prompt' not in ds.column_names:
        raise ValueError("Dataset must contain column 'prompt'")
    return ds

def load_existing_data(file_path: str) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, str]]]:
    """
    載入過往生成的數據，並建立一個集合用於快速去重。
    """
    existing_data: List[Dict[str, Any]] = []
    dedup_set: Set[Tuple[str, str]] = set()
    
    # 第一次運行時，檔案可能不存在，這是正常的
    if not os.path.exists(file_path):
        print(f"警告: 找不到檔案 {file_path}，將從零開始生成。")
        return existing_data, dedup_set

    print(f"--- 步驟 1: 讀取現有評分數據 {file_path} (保持原檔案不變) ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Existing Data"):
            try:
                item = json.loads(line)
                if all(k in item for k in ['prompt', 'response', 'reward']):
                    existing_data.append(item)
                    dedup_set.add((item['prompt'], item['response']))
            except json.JSONDecodeError:
                continue
    
    print(f"已載入 {len(existing_data)} 筆現有記錄，其中包含 {len(dedup_set)} 個唯一組合。")
    return existing_data, dedup_set

def run_rewrite_model(prompt: str, model, tokenizer, max_new_tokens=512) -> str:
    """用 Seq2Seq 模型重寫提示。"""
    instruction_prefix = "TASK: Rewrite the unsafe INPUT prompt below. The goal is to transform the potentially harmful, malicious, or unsafe input into a less harmful and safe query. The modified query MUST strictly retain the user's original core goal and intent, changing only the elements necessary for safety. Prioritize minimizing the deviation from the original context. Do not add external context or introduce new goals. The output must be the modified prompt directly and nothing else. INPUT: "
    input_text = instruction_prefix + prompt
    
    encoded = tokenizer(input_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    outputs = model.generate(
        input_ids=encoded["input_ids"].to(model.device),
        attention_mask=encoded["attention_mask"].to(model.device),
        max_length=max_new_tokens,
        do_sample=True,
        top_p=0.8,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rewritten_text.strip()


def score_and_record(prompt: str, rewritten: str, iteration_idx: int, sample_id: int) -> Dict:
    """使用 judge() 評分，並格式化結果。"""
    eval_result = judge(rewritten, prompt) 

    reward = float(eval_result["safety_score"]) * float(eval_result["relevance_score"])

    return {
        "id": sample_id,          
        "iteration": iteration_idx, 
        "prompt": prompt,
        "response": rewritten,
        "reward": reward,
        "safety": float(eval_result["safety_score"]),
        "relevance": float(eval_result["relevance_score"])
    }


def main():
    args = parse_args()

    # 1. 載入過往數據並建立去重集合 (從指定的 input 檔案讀取)
    existing_data, dedup_set = load_existing_data(args.input_scored)

    # 2. 初始化所有模型
    print("--- 步驟 2: 初始化評估與重寫模型 ---")
    initialize_models(args.guard_model, args.usefulness_model, args.chat_model)
    
    rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.rewrite_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    rewrite_tokenizer = AutoTokenizer.from_pretrained(args.rewrite_model)
    if args.lora_path is not None:
        rewrite_model = PeftModel.from_pretrained(rewrite_model, args.lora_path)
    # 3. 載入原始資料集
    ds = load_original_dataset(args.dataset)

    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    newly_generated_data: List[Dict] = [] 
    
    print("--- 步驟 3: 生成、去重並評分 (Scoring) ---")
    pbar = tqdm(total=len(ds)*args.iterations, ncols=0)
    # 4. 生成、去重和評分循環
    for idx, row in enumerate(ds):
        prompt = row["prompt"]
        
        for it in range(args.iterations):
            rewritten = run_rewrite_model(prompt, rewrite_model, rewrite_tokenizer)
            dedup_key = (prompt, rewritten)
            # 去重檢查
            if dedup_key in dedup_set:
                pbar.update(1)
                continue 
            
            # 評分並記錄新的數據
            result = score_and_record(prompt, rewritten, it, idx)
            newly_generated_data.append(result)
            
            # 將新生成的數據加入去重集合，以防在同一輪中重複
            dedup_set.add(dedup_key)
            pbar.update(1)
    print(f"\n✅ 完成生成。本次新增了 {len(newly_generated_data)} 個新的評分樣本。")
    
    # 5. 合併並儲存結果 (寫入到新的 output 檔案)
    
    output_path = args.output_scored
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # 結合舊數據和新數據
    final_data = existing_data + newly_generated_data
    final_data.sort(key=lambda x: x['id']) 
    print(f"--- 步驟 4: 儲存完整數據 ({len(final_data)} 筆記錄) 到 {output_path} ---")

    # 使用 'w' 模式 (Write/Overwrite)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in tqdm(final_data):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n🔥 評分腳本運行完成。原檔案 {args.input_scored} 未更動。")
    print(f"下一階段：運行 create_preference_pairs.py 將 {output_path} 轉換為偏好對。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n致命錯誤: {e}")
        print("請檢查您的環境設置和 'src.eval' 模組是否正確定義。") 
