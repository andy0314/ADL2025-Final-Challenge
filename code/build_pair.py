import json
import argparse
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from itertools import combinations
import os

DEFAULT_INPUT_FILE = "scored_responses_v1.jsonl" # 更新為新的預設輸出檔案名
DEFAULT_OUTPUT_FILE = "reward_dataset_pairs.jsonl"

def parse_args():
    """解析命令行參數，設定輸入和輸出檔案路徑。"""
    parser = argparse.ArgumentParser("Create Preference Pairs from Scored Responses")

    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE,
                        help="輸入檔案：包含單一評分 (prompt, response, reward) 的 JSON Lines 檔案。")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="輸出檔案：包含偏好對 (prompt, chosen, rejected, scores) 的 JSON Lines 檔案。")

    return parser.parse_args()


def load_scored_responses(file_path: str) -> List[Dict[str, Any]]:
    """
    從 JSON Lines 檔案中載入所有評分數據。
    """
    scored_responses: List[Dict[str, Any]] = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"錯誤：找不到輸入檔案 '{file_path}'。請先運行 score_rewritten_prompts.py")

    print(f"--- 步驟 1: 讀取單一評分數據 {file_path} ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Scored Data"):
            try:
                item = json.loads(line)
                # 僅保留核心欄位
                if all(k in item for k in ['prompt', 'response', 'reward']):
                    scored_responses.append({
                        'prompt': item['prompt'],
                        'response': item['response'],
                        'reward': item['reward']
                    })
            except json.JSONDecodeError:
                continue
    
    print(f"成功載入 {len(scored_responses)} 筆評分記錄。")
    return scored_responses


def create_preference_pairs(raw_data: List[Dict]) -> List[Dict]:
    """
    將單一評分的資料轉換為 (prompt, chosen_response, rejected_response, scores) 的偏好對。
    """
    
    # 1. 根據 prompt 內容分組
    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    for item in raw_data:
        prompt = item['prompt']
        if prompt not in grouped_data:
            grouped_data[prompt] = []
        # 儲存 {response 內容, reward 分數}
        grouped_data[prompt].append({'response': item['response'], 'reward': item['reward']})

    preference_pairs: List[Dict] = []
    
    # 2. 對每個 prompt 組建立偏好對
    print("\n--- 步驟 2: 建立偏好對 (Preference Pairs) ---")
    for prompt, response_list in tqdm(grouped_data.items(), desc="Creating Pairs"):
        if len(response_list) < 2:
            continue

        # 使用 itertools.combinations 產生所有可能的兩兩組合
        for res_a, res_b in combinations(response_list, 2):
            reward_a = res_a['reward']
            reward_b = res_b['reward']

            # 只有在 reward 分數不相等時才形成有效的偏好對
            if reward_a != reward_b:
                if reward_a > reward_b:
                    chosen = res_a['response']
                    rejected = res_b['response']
                    chosen_score = reward_a
                    rejected_score = reward_b
                else:
                    chosen = res_b['response']
                    rejected = res_a['response']
                    chosen_score = reward_b
                    rejected_score = reward_a
                    
                preference_pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_score": chosen_score, 
                    "rejected_score": rejected_score 
                })

    return preference_pairs


def main():
    args = parse_args()

    try:
        # 1. 載入單一評分數據
        scored_data = load_scored_responses(args.input)
        
        # 2. 轉換為偏好對
        preference_pairs = create_preference_pairs(scored_data)
        
        # 3. 儲存結果
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        print(f"\n✅ 成功創建 {len(preference_pairs)} 個偏好對。")
        print(f"--- 步驟 3: 儲存數據到 {output_path} ---")

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in tqdm(preference_pairs, desc="Writing Pairs"):
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        print(f"\n🔥 獎勵模型數據集 (Preference Pairs) 已創建完成。")

    except FileNotFoundError as e:
        print(f"錯誤: {e}")
    except Exception as e:
        print(f"腳本執行失敗: {e}")


if __name__ == "__main__":
    main()
