import json
import argparse
from typing import Set
import os

def parse_args():
    """
    解析命令行參數，設定資料集檔案路徑。
    """
    parser = argparse.ArgumentParser(description="計算 JSON Lines 資料集中唯一 Prompt 的數量。")
    parser.add_argument("--dataset_path", 
                        type=str, 
                        default="reward_dataset_pairs.jsonl",
                        help="輸入的 JSON Lines 檔案路徑 (例如: reward_dataset_pairs.jsonl)。")
    return parser.parse_args()

def count_unique_prompts(file_path: str) -> Set[str]:
    """
    從 JSON Lines 檔案中讀取並計算所有唯一的 'prompt' 欄位值。

    Args:
        file_path (str): JSON Lines 檔案的路徑。

    Returns:
        Set[str]: 包含所有唯一 Prompt 字串的集合。
    """
    unique_prompts: Set[str] = set()
    total_lines = 0
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"錯誤：找不到指定的檔案 '{file_path}'。")

    print(f"--- 開始處理檔案: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        # 假設 'prompt' 欄位存在且是字串
                        if 'prompt' in record and isinstance(record['prompt'], str):
                            unique_prompts.add(record['prompt'])
                        else:
                            print(f"警告：第 {line_num} 行遺失 'prompt' 欄位或格式不正確。")
                    except json.JSONDecodeError as e:
                        print(f"錯誤：解析 JSON 失敗 (第 {line_num} 行)。請檢查檔案格式。錯誤訊息: {e}")
                        continue
    except Exception as e:
        print(f"讀取檔案時發生意外錯誤: {e}")
        return set() # 發生錯誤時返回空集合

    print(f"總共處理了 {total_lines} 行數據。")
    return unique_prompts

def main():
    args = parse_args()
    
    try:
        unique_set = count_unique_prompts(args.dataset_path)
        
        if unique_set:
            print("--- 結果 ---")
            print(f"總記錄行數: (已在上方顯示)")
            print(f"🎉 唯一未修改的 Prompt 數量: {len(unique_set)}")
        else:
            print("未能找到任何唯一的 Prompt。請檢查檔案內容和格式。")

    except FileNotFoundError as e:
        print(f"{e}")
    except Exception as e:
        print(f"腳本執行失敗: {e}")


if __name__ == "__main__":
    main()
