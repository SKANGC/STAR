import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
from collections import Counter
import json
import time
from argparse import ArgumentParser
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
import h5py
import gc
import psutil
from sklearn.model_selection import train_test_split

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# 内存监控函数
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB"

# 模型路径配置 微调后的模型路径
model_path = "./final_model"

# 优化模型加载
qwen_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

def generate_embeddings_optimized(logs, devices='BGL', batch_size=16):
    """
    优化的嵌入生成函数：返回 2D numpy 数组，每 batch 直接生成 embeddings
    """
    all_embeddings = []
    system_prompt = (
        "你是一个安全分析助手，需要为进程信息生成向量表示，重点关注进程日志"
        "请综合分析以下信息：\n"
        "1. 进程运行时日志\n"
        "2. 运行环境\n"
    )

    with torch.no_grad():
        for i in range(0, len(logs), batch_size):
            gc.collect()
            torch.cuda.empty_cache()

            batch_logs = logs[i:i+batch_size]

            formatted_texts = []
            for log in batch_logs:
                user_input = (
                    f"### 进程日志:\n{log}\n\n"
                    f"### 运行设备:\n{devices}\n\n"
                    "请综合分析上述信息生成嵌入表示"
                )
                formatted_texts.append(
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_input}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

            inputs = tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_token_type_ids=False
            ).to(qwen_model.device)

            outputs = qwen_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].mean(dim=1)
            batch_embeddings_np = hidden_states.to(torch.float32).cpu().numpy().astype(np.float16)

            all_embeddings.append(batch_embeddings_np)

            del inputs, outputs, hidden_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if i % (batch_size * 4) == 0:
                gc.collect()
                print(f"Processed {i}/{len(logs)} samples. {memory_usage()}")

    # 拼接所有 batch
    embeddings_np = np.vstack(all_embeddings)  # shape: (total_samples, hidden_dim)
    return embeddings_np

def save_embeddings_incrementally_h5(
        embeddings_batch, labels_batch, contents_batch, templates_batch,
        file_prefix, batch_index, total_batches, mode='append'):
    """
    增量保存嵌入向量和相关元数据到 HDF5 文件。
    """
    os.makedirs(os.path.dirname(file_prefix), exist_ok=True)
    h5_file = file_prefix + '.h5'

    # 确保 embeddings 是 2D numpy
    embeddings_batch = np.asarray(embeddings_batch, dtype=np.float16)
    # 将列表转为 1D numpy string
    labels_np = np.array(labels_batch, dtype='S')
    contents_np = np.array(contents_batch, dtype='S')
    templates_np = np.array(templates_batch, dtype='S')

    if mode == 'new' or not os.path.exists(h5_file):
        # 新建文件
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset(
                'embeddings', data=embeddings_batch,
                maxshape=(None, embeddings_batch.shape[1]), chunks=True
            )
            f.create_dataset('labels', data=labels_np, maxshape=(None,), chunks=True)
            f.create_dataset('contents', data=contents_np, maxshape=(None,), chunks=True)
            f.create_dataset('templates', data=templates_np, maxshape=(None,), chunks=True)
    else:
        # 追加模式
        with h5py.File(h5_file, 'a') as f:
            # embeddings
            dset = f['embeddings']
            old_size = dset.shape[0]
            new_size = old_size + embeddings_batch.shape[0]
            dset.resize((new_size, dset.shape[1]))
            dset[old_size:] = embeddings_batch

            # labels
            dset = f['labels']
            old_size = dset.shape[0]
            new_size = old_size + labels_np.shape[0]
            dset.resize((new_size,))
            dset[old_size:] = labels_np

            # contents
            dset = f['contents']
            old_size = dset.shape[0]
            new_size = old_size + contents_np.shape[0]
            dset.resize((new_size,))
            dset[old_size:] = contents_np

            # templates
            dset = f['templates']
            old_size = dset.shape[0]
            new_size = old_size + templates_np.shape[0]
            dset.resize((new_size,))
            dset[old_size:] = templates_np

    print(f"Batch {batch_index + 1}/{total_batches} saved to {h5_file}")
    return h5_file

def load_embeddings_from_h5(file_prefix):
    """
    从 HDF5 文件加载 embeddings 和元数据。
    
    返回：
        embeddings: np.ndarray, shape=(N, hidden_dim), dtype=float16
        labels: list[str]
        contents: list[str]
        templates: list[str]
    如果文件不存在，则返回 None, None, None, None
    """
    h5_file = file_prefix + '.h5'
    if not os.path.exists(h5_file):
        return None, None, None, None

    with h5py.File(h5_file, 'r') as f:
        embeddings = np.array(f['embeddings'], dtype=np.float16)
        labels = [s.decode('utf-8') for s in f['labels']]  # bytes -> str
        contents = [s.decode('utf-8') for s in f['contents']]
        templates = [s.decode('utf-8') for s in f['templates']]

    print(f"Loaded embeddings from {h5_file}, shape: {embeddings.shape}")
    print(f"Labels: {len(labels)}, Contents: {len(contents)}, Templates: {len(templates)}")
    return embeddings, labels, contents, templates

def extract_process_feature_from_data_optimized(data_dict, embeddings_dir, dataset_type, 
                                                force_regenerate=False, chunk_size=1000):
    """
    优化的特征提取函数：分批生成和保存嵌入向量，每 batch 直接保存到 HDF5
    增加功能：
        若检测到已有 HDF5 文件，则读取其中样本数量，
        若未覆盖全部样本，则从中断处继续生成。
    """
    global qwen_model, tokenizer
    file_prefix = os.path.join(embeddings_dir, f"{dataset_type}_embeddings")
    h5_file = file_prefix + ".h5"

    total_samples = len(data_dict)
    total_chunks = (total_samples + chunk_size - 1) // chunk_size

    # 检查是否已存在嵌入文件
    existing_count = 0
    if os.path.exists(h5_file) and not force_regenerate:
        try:
            with h5py.File(h5_file, "r") as f:
                if "embeddings" in f:
                    existing_count = f["embeddings"].shape[0]
                    print(f"[INFO] Found existing embeddings file: {h5_file}")
                    print(f"[INFO] Existing embeddings: {existing_count}/{total_samples}")
        except Exception as e:
            print(f"[WARN] Failed to read existing HDF5 file: {repr(e)}")
            existing_count = 0

    # 若已有全部样本嵌入则直接返回
    if existing_count >= total_samples and not force_regenerate:
        print(f"[INFO] All {total_samples} embeddings already exist. Loading from HDF5...")
        embeddings, labels, contents, templates = load_embeddings_from_h5(file_prefix)
        process_data = {
            idx: {
                'features': embeddings[idx].tolist(),
                'label': labels[idx],
                'content': contents[idx],
                'template': templates[idx]
            } for idx in range(len(embeddings))
        }
        return process_data

    # 若部分样本已存在，从中断处继续生成
    start_chunk = existing_count // chunk_size
    if existing_count > 0:
        print(f"[RESUME] Resuming embedding generation from chunk {start_chunk + 1} "
              f"(skipping first {existing_count} samples).")

    print(f"Generating new embeddings for {dataset_type} in chunks (total {total_chunks})...")

    for chunk_idx in range(start_chunk, total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
        chunk_data = data_dict[start_idx:end_idx]

        print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks} (samples {start_idx}-{end_idx})...")
        print(f"Current memory usage: {memory_usage()}")

        # 提取当前 chunk 数据
        chunk_logs = [row['Content'] for row in chunk_data]
        chunk_contents = [row['Content'] for row in chunk_data]
        chunk_labels = [row['label'] for row in chunk_data]

        # 生成当前 chunk embeddings
        chunk_embeddings = generate_embeddings_optimized(chunk_logs)

        # 立即保存当前 chunk 到 HDF5
        save_embeddings_incrementally_h5(
            chunk_embeddings, chunk_labels, chunk_contents, chunk_templates,
            file_prefix, chunk_idx, total_chunks,
            mode='append' if (os.path.exists(h5_file) or chunk_idx > 0) else 'new'
        )

        # 清理显存和内存
        del chunk_logs, chunk_templates, chunk_embeddings, chunk_contents
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Chunk {chunk_idx + 1} completed and saved. {memory_usage()}")

        # 定期清理并重载模型
        if (chunk_idx + 1) % 200 == 0:
            print(f"[INFO] Forcing cleanup after {chunk_idx+1} chunks")

            try:
                del qwen_model
                del tokenizer
                torch.cuda.empty_cache()
                gc.collect()
                print("[CLEANUP] Model and tokenizer deleted. GPU memory cleared.")
            except Exception as e:
                print(f"[WARN] Cleanup failed: {repr(e)}")

            time.sleep(30)

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                )
                print("[RELOAD] Model and tokenizer reloaded successfully.")
            except Exception as e:
                print(f"[FATAL] Model reloading failed: {repr(e)}")
                raise e

    # 加载完整 embeddings 构建 process_data
    embeddings, labels, contents, templates = load_embeddings_from_h5(file_prefix)
    process_data = {
        idx: {
            'features': embeddings[idx].tolist(),
            'label': labels[idx],
            'content': contents[idx],
            'template': templates[idx]
        } for idx in range(len(embeddings))
    }

    del embeddings, labels, contents, templates
    gc.collect()

    print(f"Generated embeddings for {len(process_data)} samples")
    return process_data

def create_model_directories(dataset_path):
    """创建保存模型和结果的目录结构"""
    file_over = "lora"
    models_dir = os.path.join(dataset_path, f'{file_over}/trained_models')
    results_dir = os.path.join(dataset_path, f'{file_over}/results')
    errors_dir = os.path.join(dataset_path, f'{file_over}/error_analysis')
    data_dir = os.path.join(dataset_path, 'data')
    embeddings_dir = os.path.join(dataset_path, f'{file_over}/embeddings')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    return models_dir, results_dir, errors_dir, data_dir, embeddings_dir

def split_and_save_raw_data(dataset_path, test_size=0.3, random_state=42, force_resplit=False):
    """
    从单个大CSV文件中划分原始数据并保存到本地
    """
    data_dir = os.path.join(dataset_path, 'data')
    
    # 检查是否已存在划分文件
    train_data_file = os.path.join(data_dir, 'train.csv')
    test_data_file = os.path.join(data_dir, 'test.csv')
    
    if os.path.exists(train_data_file) and os.path.exists(test_data_file) and not force_resplit:
        print(train_data_file)
        print(test_data_file)
        print("Split data already exists. Loading from local files...")
        return train_data_file, test_data_file
    
    print("Splitting raw data from single large CSV file...")
    
    main_csv_file = os.path.join("/root/sc_code/bgl/all_matched_results.csv")
    
    # 读取整个数据集
    all_data = []
    total_rows = 0
    
    # 使用分块读取来处理大文件
    chunk_size = 10000  # 每次读取10000行
    for chunk in pd.read_csv(main_csv_file, chunksize=chunk_size, encoding='utf-8'):
        # 将每个chunk转换为字典列表
        chunk_data = chunk.to_dict('records')
        all_data.extend(chunk_data)
        total_rows += len(chunk_data)
        print(f"Loaded {total_rows} rows...")
    
    print(f"Total samples loaded: {len(all_data)}")
    
    # 验证数据格式
    if not all_data:
        raise ValueError("No data loaded from CSV file")
    
    # 检查必需的列
    required_columns = ['Content', 'EventTemplate', 'label']
    missing_columns = [col for col in required_columns if col not in all_data[0]]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")
    
    # 确保label列存在且是整数类型
    for row in all_data:
        if 'label' not in row:
            print(f"Warning: row missing 'label' column: {row}")
            row['label'] = 0  # 默认设为正常
        else:
            try:
                row['label'] = int(row['label'])
            except (ValueError, TypeError):
                print(f"Warning: invalid label value '{row['label']}', setting to 0")
                row['label'] = 0
    
    # 统计标签分布
    label_counts = {}
    for row in all_data:
        label = row['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Label distribution in dataset:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count} samples ({count/len(all_data)*100:.2f}%)")
    
    # 提取标签用于分层抽样
    labels = [row['label'] for row in all_data]
    
    # 划分训练集和测试集
    print("Splitting data into train and test sets...")
    train_indices, test_indices = train_test_split(
        range(len(all_data)), 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    # 创建训练集和测试集数据
    train_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # 保存训练集
    if train_data:
        with open(train_data_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = list(train_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # 分批写入以避免内存问题
            for i in range(0, len(train_data), chunk_size):
                batch = train_data[i:i+chunk_size]
                writer.writerows(batch)
                print(f"Saved train batch {i//chunk_size + 1}")
    
    # 保存测试集
    if test_data:
        with open(test_data_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = list(test_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # 分批写入以避免内存问题
            for i in range(0, len(test_data), chunk_size):
                batch = test_data[i:i+chunk_size]
                writer.writerows(batch)
                print(f"Saved test batch {i//chunk_size + 1}")
    
    # 保存数据统计信息
    train_labels = [row['label'] for row in train_data]
    test_labels = [row['label'] for row in test_data]
    
    stats = {
        'source_file': main_csv_file,
        'total_samples': len(all_data),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'train_label_distribution': dict(Counter(train_labels)),
        'test_label_distribution': dict(Counter(test_labels)),
        'split_ratio': test_size,
        'random_state': random_state,
        'split_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    stats_file = os.path.join(data_dir, 'data_split_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Raw split data saved to: {data_dir}")
    print(f"Train data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    print(f"Train label distribution: {stats['train_label_distribution']}")
    print(f"Test label distribution: {stats['test_label_distribution']}")
    
    # 清理内存
    del all_data, train_data, test_data
    gc.collect()
    
    return train_data_file, test_data_file

def load_and_process_split_data(train_data_file, test_data_file, embeddings_dir, force_regenerate=False):
    """
    从本地文件加载划分好的训练集和测试集，并处理为特征
    """
    print("Loading and processing split data from local files...")
    
    # 加载训练集
    train_data = []
    with open(train_data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['label'] = int(row['label'])
            train_data.append(row)
    
    # 加载测试集
    test_data = []
    with open(test_data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['label'] = int(row['label'])
            test_data.append(row)
    
    print(f"Loaded train set: {len(train_data)} samples")
    print(f"Loaded test set: {len(test_data)} samples")
    
    # 处理训练集特征
    print("Processing training data features...")
    train_processed = extract_process_feature_from_data_optimized(
        train_data, embeddings_dir, "train", force_regenerate, chunk_size=128
    )
    
    # 处理测试集特征
    print("Processing test data features...")
    test_processed = extract_process_feature_from_data_optimized(
        test_data, embeddings_dir, "test", force_regenerate, chunk_size=128
    )
    
    # 准备训练数据
    X_train = []
    y_train = []
    train_contents = []
    train_templates = []
    train_sample_ids = []
    
    for sample_id, data in train_processed.items():
        X_train.append(data['features'])
        y_train.append(data['label'])
        train_contents.append(data['content'])
        train_templates.append(data['template'])
        train_sample_ids.append(sample_id)
    
    # 准备测试数据
    X_test = []
    y_test = []
    test_contents = []
    test_templates = []
    test_sample_ids = []
    
    for sample_id, data in test_processed.items():
        X_test.append(data['features'])
        y_test.append(data['label'])
        test_contents.append(data['content'])
        test_templates.append(data['template'])
        test_sample_ids.append(sample_id)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Processed train set: {X_train.shape}")
    print(f"Processed test set: {X_test.shape}")
    print(f"Train anomalies: {sum(y_train)}, Test anomalies: {sum(y_test)}")
    
    return (X_train, y_train, train_contents, train_templates, train_sample_ids,
            X_test, y_test, test_contents, test_templates, test_sample_ids)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--d", type=str, default='./bgl')
    parser.add_argument("--resplit", action="store_true", help="Force resplit data even if split files exist")
    parser.add_argument("--regen_embeddings", action="store_true", help="Force regenerate embeddings even if they exist")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for incremental embedding generation")
    args = parser.parse_args()
    dataset = args.d
    force_resplit = args.resplit
    force_regenerate_embeddings = args.regen_embeddings
    chunk_size = args.chunk_size
    
    # 检查资源状态
    print("=" * 50)
    print("EMBEDDING GENERATION - RESOURCE STATUS CHECK")
    print("=" * 50)
    print(memory_usage())
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 50)
    
    # 创建保存目录
    models_dir, results_dir, errors_dir, data_dir, embeddings_dir = create_model_directories(dataset)
    
    # 提前划分并保存原始数据
    print("\n" + "="*50)
    print("RAW DATA SPLITTING PHASE")
    print("="*50)
    train_data_file, test_data_file = split_and_save_raw_data(
        dataset, test_size=0.3, random_state=42, force_resplit=force_resplit
    )
    
    # 从本地文件加载划分好的数据并生成嵌入
    print("\n" + "="*50)
    print("EMBEDDING GENERATION PHASE")
    print("="*50)
    (X_train, y_train, train_contents, train_templates, train_sample_ids,
     X_test, y_test, test_contents, test_templates, test_sample_ids) = load_and_process_split_data(
        train_data_file, test_data_file, embeddings_dir, force_regenerate_embeddings
    )
    
    print(f"Memory after embedding generation: {memory_usage()}")
    
    # 最终总结
    print(f"\n{'='*60}")
    print("EMBEDDING GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Train embeddings: {X_train.shape}")
    print(f"Test embeddings: {X_test.shape}")
    print(f"Train anomalies: {sum(y_train)}")
    print(f"Test anomalies: {sum(y_test)}")
    print(f"Embeddings saved to: {embeddings_dir}")
    print(f"Final memory usage: {memory_usage()}")
    print(f"{'='*60}")
    
    # 保存数据统计信息
    stats = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_anomalies': int(sum(y_train)),
        'test_anomalies': int(sum(y_test)),
        'train_normal': int(len(y_train) - sum(y_train)),
        'test_normal': int(len(y_test) - sum(y_test)),
        'embedding_dim': X_train.shape[1],
        'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    stats_file = os.path.join(embeddings_dir, 'embedding_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Embedding statistics saved to: {stats_file}")