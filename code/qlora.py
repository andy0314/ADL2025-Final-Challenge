import torch
import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    logging,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from typing import Dict, List, Any

# 導入 PEFT 相關函式
from peft import LoraConfig, PeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator 

logging.set_verbosity_error()

def parse_args():
    """解析命令行參數，設定模型和訓練配置。"""
    parser = argparse.ArgumentParser(description="LoRA SFT Fine-tuning script for Prompt Rewriter Model.")
    
    # --- Model and Path Arguments ---
    # 將預設模型更改為 Flan-T5-XL 以展示 LoRA 的優勢
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large", 
                        help="要微調的基礎模型名稱 (例如: google/flan-t5-xl)。")
    parser.add_argument("--dataset_path", type=str, default="./data2/reward_dataset.jsonl",
                        help="包含 prompt 和 chosen 回應的 JSON Lines 檔案路徑。")
    parser.add_argument("--output_dir", type=str, default="./models/lora_1",
                        help="LoRA 權重和配置儲存的目錄。")
    parser.add_argument("--max_length", type=int, default=512,
                        help="分詞的最大序列長度。")
    parser.add_argument("--lora_path", type=str, default=None)
    # --- Training Arguments ---
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="訓練輪數 (Epochs)。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, # LoRA 通常使用更高的學習率
                        help="優化器的學習率。")
    parser.add_argument("--batch_size", type=int, default=2, # LoRA 可以使用更大的批次大小
                        help="每個設備的訓練批次大小 (Batch Size)。")
    
    # --- LoRA Arguments ---
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA 秩 (rank)。")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA 縮放因子 (scaling factor)。")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA 層的 Dropout 機率。")
    
    return parser.parse_args()

def load_and_tokenize_data(tokenizer: AutoTokenizer, dataset_path: str, max_length: int) -> load_dataset:
    """載入資料集，並只保留 'prompt' 和 'chosen' 欄位進行分詞。"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}.")
    
    print(f"Loading dataset from {dataset_path}...")
    data_files = {"train": dataset_path}
    dataset = load_dataset("json", data_files=data_files)["train"]

    if not all(col in dataset.column_names for col in ['prompt', 'chosen']):
        raise ValueError("Dataset must contain 'prompt' and 'chosen' columns.")

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, List[int]]:
        """將文本轉換為 tokens，並設置標籤 (labels)。"""
        instruction_prefix = "TASK: Rewrite the unsafe INPUT prompt below. The goal is to transform the potentially harmful, malicious, or unsafe input into a less harmful and safe query. The modified query MUST strictly retain the user's original core goal and intent, changing only the elements necessary for safety. Prioritize minimizing the deviation from the original context. Do not add external context or introduce new goals. The output must be the modified prompt directly and nothing else. INPUT: "

        # 1. 輸入 (Prompt) 分詞
        inputs = [instruction_prefix + p for p in examples["prompt"]]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

        # 2. 標籤 (Chosen Response) 分詞
        # 移除 with tokenizer.as_target_tokenizer():
        # 由於 T5 的 Encoder-Decoder 結構，我們直接對目標文本進行分詞
        labels = tokenizer(examples["chosen"], max_length=max_length, truncation=True)

        # 將解碼器輸入 (labels) 附加到編碼器輸入中
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    # 增加 num_proc 以加快分詞速度
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )
    return tokenized_datasets

def train_sft_peft(args):
    accelerator = Accelerator()
    device = accelerator.device
    
    # --- 1. 載入模型和分詞器 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model_name}...")
    
    # 設置 BFLOAT16 精度以節省 VRAM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        # 僅在單卡訓練時使用 device_map="auto"；使用 Accelerator/DDP 時通常不設置
        # device_map="auto" 
    )

    # 針對 T5 模型的 PEFT 配置
    target_modules = ["q", "v"] # T5 模型通常在 'q', 'v' (Query, Value) 矩陣上應用 LoRA
    
    # --- 2. 應用 LoRA 配置 ---
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    # 將 LoRA 模塊添加到基礎模型中
    if args.lora_path != None:
        print("Load old LoRA weight")
        model = PeftModel.from_pretrained(model, args.lora_path)

    else:
        print("Load new LoRA weight")
        model = get_peft_model(model, peft_config)
    for name, param in model.named_parameters():
        if 'lora' not in name:
            # 凍結所有非 LoRA 權重
            param.requires_grad = False
        else:
            # 確保所有 LoRA 權重是可訓練的
            param.requires_grad = True
    print("--- LoRA Model Configuration ---")
    model.print_trainable_parameters() # 顯示可訓練參數的數量
    print("--------------------------------")

    # --- 3. 載入並預處理資料 ---
    train_dataset = load_and_tokenize_data(
        tokenizer,
        args.dataset_path,
        args.max_length
    )

    # --- 4. 配置 PEFT 訓練參數 ---
    # 由於使用 LoRA，我們可以增大批次大小 (Batch Size)
    optimizer_type = "adamw_torch" 

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4, # 由於 batch_size 增大，可減少累積步驟
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        optim=optimizer_type,
        lr_scheduler_type="cosine",
        save_strategy="epoch",  # 每 epoch 保存一次 LoRA 權重
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=50,
        resume_from_checkpoint=False,
        remove_unused_columns=True,
        bf16=True, # 必須使用 BF16 來訓練大型模型
        load_best_model_at_end=False,
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 5. 啟動訓練 ---
    print("\nStarting LoRA SFT fine-tuning on Chosen responses...")
    trainer.train()

    # --- 6. 儲存 LoRA 權重 ---
    final_lora_path = args.output_dir
    trainer.model.save_pretrained(final_lora_path)
    tokenizer.save_pretrained(final_lora_path)
    print(f"\n✅ LoRA PEFT Fine-tuning finished. Weights saved to {final_lora_path}")

    # 💡 要將 LoRA 權重與原始模型合併以進行推論，請參考以下 LoRA 權重合併步驟。
    # 這是可選的，直接使用 peft_model.to_infer() 進行推論也是可以的。


if __name__ == "__main__":
    try:
        args = parse_args()
        train_sft_peft(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("請確認您的數據集檔案路徑正確。")
    except Exception as e:
        print(f"An error occurred during PEFT training: {e}")
        print("請檢查您的環境設置 (例如: PyTorch, Accelerate, PEFT 版本) 和 GPU 配置。")
