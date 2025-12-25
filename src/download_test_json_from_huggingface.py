import requests
import json
import os

# --- 脚本配置 ---
DATASET_NAME = "Jiann/STORAL"
CONFIG_NAME = "default"
SPLIT_NAME = "storal_zh_train" # 假设您需要 zh_train 划分的数据
LENGTH = 100 # 单次请求的最大长度
TOTAL_ROWS = 200 # 目标获取总行数
OUTPUT_FILE = "storal_zh_200_rows.json"
BASE_URL = "https://datasets-server.huggingface.co/rows"
# -----------------

def fetch_data_chunk(offset, length):
    """从 Hugging Face 数据集服务器获取一个数据块。"""
    params = {
        "dataset": DATASET_NAME,
        "config": CONFIG_NAME,
        "split": SPLIT_NAME,
        "offset": offset,
        "length": length
    }
    print(f"-> 正在请求数据： offset={offset}, length={length}")
    
    try:
        # 发送 GET 请求
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status() # 如果状态码不是 200，则抛出异常
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"错误：请求失败：{e}")
        # 如果请求失败，返回 None 或空对象
        return {"rows": []} 

def main():
    # 最终用于存储合并数据的对象，初始化为 None
    final_data = None
    
    # 循环分批获取数据
    for start_offset in range(0, TOTAL_ROWS, LENGTH):
        chunk_data = fetch_data_chunk(start_offset, LENGTH)
        
        # 确保数据块包含 'rows' 键
        if "rows" in chunk_data:
            if final_data is None:
                # 第一次请求：初始化最终数据结构
                final_data = chunk_data.copy()
            else:
                # 后续请求：合并 'rows' 数组
                final_data['rows'].extend(chunk_data['rows'])
        else:
            print(f"警告：offset={start_offset} 的请求没有返回 'rows' 数据。")
            
    if final_data is None:
        print("致命错误：未能获取任何数据。")
        return

    # 打印最终获取的行数
    print(f"\n✅ 数据获取完成。总行数：{len(final_data['rows'])}")

    # 将合并后的数据保存为美化 (pretty print) 的 JSON 文件
    print(f"💾 正在保存到 {OUTPUT_FILE}...")
    try:
        # 使用 json.dump 进行美化保存 (indent=4)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print("✨ 保存成功！")
        
    except Exception as e:
        print(f"错误：保存文件失败：{e}")


if __name__ == "__main__":
    main()