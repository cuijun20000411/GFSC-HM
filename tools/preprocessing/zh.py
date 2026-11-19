# ===================== 【学术合规修复版】留一法纯空间预处理代码 | 完美适配特征提取 =====================
import os
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import gc
import random

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "backend": "Agg"
})

# ===================== 全局固定随机种子（100%可复现） =====================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ===================== 全局配置 =====================
class Config:
    # 原始数据根目录
    ROOT_DATA_PATH = "/etc/cj/boundary/boundary"
    # 预处理输出目录
    OUTPUT_DIR = "/etc/cj/boundary/duoyuan"
    
    # 数据参数
    IMG_SHAPE = (32, 32)
    TOTAL_GRIDS = 32 * 32
    NUM_GRIDS = 999
    
    POLLUTANTS = ["no2", "o3", "pm25"]
    
    # 完美匹配你的数据集结构 (注意 pm2_5 的文件夹命名)
    POLLUTANT_PATH = {
        "no2": {
            "label": os.path.join(ROOT_DATA_PATH, "no2", "no2_label.npy"),
            "features": os.path.join(ROOT_DATA_PATH, "no2", "no2_features.npy")
        },
        "o3": {
            "label": os.path.join(ROOT_DATA_PATH, "o3", "o3_label.npy"),
            "features": os.path.join(ROOT_DATA_PATH, "o3", "o3_features.npy")
        },
        "pm25": {
            "label": os.path.join(ROOT_DATA_PATH, "pm2_5", "pm2_5_label.npy"),
            "features": os.path.join(ROOT_DATA_PATH, "pm2_5", "pm2_5_features.npy")
        }
    }
    
    GRID_LABEL_PATH = os.path.join(ROOT_DATA_PATH, "grid_label.csv")
    ADJ_MATRIX_PATH = os.path.join(ROOT_DATA_PATH, "adj.npy")

# ===================== 路径预校验 =====================
def check_all_paths(config: Config):
    logging.info("🔍 预检查所有文件路径...")
    all_exist = True
    for pollutant, paths in config.POLLUTANT_PATH.items():
        for file_type, file_path in paths.items():
            if not os.path.exists(file_path):
                logging.error(f"❌ {pollutant} 的 {file_type} 文件不存在：{file_path}")
                all_exist = False
    
    if all_exist:
        logging.info("✅ 所有路径检查通过")
    else:
        raise FileNotFoundError("部分文件不存在，请检查路径结构")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "base_info"), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "visualization"), exist_ok=True)

# ===================== 加载站点基础信息 =====================
def load_station_info(config: Config):
    grid_label = pd.read_csv(config.GRID_LABEL_PATH)
    station_indices = grid_label.iloc[:, 1].values.astype(int)
    station_indices = station_indices[(station_indices >= 0) & (station_indices < config.NUM_GRIDS)]
    station_indices = np.unique(station_indices)
    
    station_positions, station_grid_idx = [], []
    for idx in station_indices:
        row, col = idx // config.IMG_SHAPE[1], idx % config.IMG_SHAPE[1]
        station_positions.append((row, col))
        station_grid_idx.append(idx)
    
    station_positions = np.array(station_positions, dtype=np.float32)
    station_grid_idx = np.array(station_grid_idx, dtype=int)
    non_station_grid_idx = np.array([i for i in range(config.NUM_GRIDS) if i not in station_grid_idx], dtype=int)
    
    global_mask = np.zeros(config.IMG_SHAPE, dtype=np.float32)
    for (row, col) in station_positions: 
        global_mask[int(row), int(col)] = 1.0
    
    base_info_dir = os.path.join(config.OUTPUT_DIR, "base_info")
    np.save(os.path.join(base_info_dir, "station_indices.npy"), station_indices)
    np.save(os.path.join(base_info_dir, "station_positions.npy"), station_positions)
    np.save(os.path.join(base_info_dir, "station_grid_idx.npy"), station_grid_idx)
    np.save(os.path.join(base_info_dir, "non_station_grid_idx.npy"), non_station_grid_idx)
    np.save(os.path.join(base_info_dir, "global_mask.npy"), global_mask)
    np.save(os.path.join(base_info_dir, "adj_matrix.npy"), np.load(config.ADJ_MATRIX_PATH))
    
    logging.info(f"✅ 站点信息加载完成：有效监测站 {len(station_indices)} 个")
    return station_indices, station_positions, station_grid_idx, global_mask, non_station_grid_idx

# ===================== 全量特征与浓度处理 =====================
def process_pollutant_data_loocv(config: Config, pollutant: str, station_grid_idx: np.ndarray, global_mask: np.ndarray, station_positions: np.ndarray):
    logging.info(f"\n===================== 开始处理 {pollutant.upper()} =====================")
    
    # 1. 加载浓度(label)与多源协变量特征(features)
    conc_raw_data = np.load(config.POLLUTANT_PATH[pollutant]["label"]) # (T, 999)
    features_raw_data = np.load(config.POLLUTANT_PATH[pollutant]["features"]) # (T, 999, F)
    
    T = conc_raw_data.shape[0]
    F_dim = features_raw_data.shape[-1] if len(features_raw_data.shape) == 3 else 1
    logging.info(f"✅ {pollutant.upper()}：总时间步 {T} | 提取到多源稠密特征数：{F_dim} 维")
    
    # 2. 严格时序拆分，杜绝泄露
    train_split = int(T * 0.7)
    val_split = int(T * 0.85)
    train_indices_full = np.arange(train_split)
    val_indices_full = np.arange(train_split, val_split)
    test_indices_full = np.arange(val_split, T)
    
    # 3. 筛选全站点有有效浓度的时间步
    train_conc_raw = conc_raw_data[train_indices_full]
    valid_train_indices = train_indices_full[np.all(np.isfinite(train_conc_raw[:, station_grid_idx]), axis=1)]
    valid_val_indices = val_indices_full[np.all(np.isfinite(conc_raw_data[val_indices_full][:, station_grid_idx]), axis=1)]
    valid_test_indices = test_indices_full[np.all(np.isfinite(conc_raw_data[test_indices_full][:, station_grid_idx]), axis=1)]
    
    valid_time_indices = np.concatenate([valid_train_indices, valid_val_indices, valid_test_indices])
    T_valid = len(valid_time_indices)
    logging.info(f"✅ 有效时间步对齐完成，共保留 {T_valid} 个时间步")
    
    # 4. 拟合浓度 Scaler (仅使用训练集的有效站点数据)
    valid_train_vals = conc_raw_data[valid_train_indices][:, station_grid_idx].reshape(-1, 1)
    conc_scaler = StandardScaler()
    conc_scaler.fit(valid_train_vals)
    conc_raw_norm = (conc_raw_data - conc_scaler.mean_[0]) / (conc_scaler.scale_[0] + 1e-8)
    
    train_conc_norm = conc_raw_norm[valid_train_indices]
    fixed_global_mean = np.nanmean(train_conc_norm[:, station_grid_idx])
    fixed_global_std = np.nanstd(train_conc_norm[:, station_grid_idx])
    
    # 5. 【核心】：拟合多源特征 Scaler (同样仅用训练集，彻底防泄露)
    train_features = features_raw_data[valid_train_indices] 
    feat_scaler = StandardScaler()
    feat_scaler.fit(train_features.reshape(-1, F_dim))
    features_norm = feat_scaler.transform(features_raw_data.reshape(-1, F_dim)).reshape(features_raw_data.shape)
    
    # 6. 映射生成 32x32 网格图像
    coord_img = np.zeros((config.IMG_SHAPE[0], config.IMG_SHAPE[1], 2), dtype=np.float32)
    for r in range(config.IMG_SHAPE[0]):
        for c in range(config.IMG_SHAPE[1]):
            coord_img[r, c] = [r / config.IMG_SHAPE[0], c / config.IMG_SHAPE[1]]
            
    conc_img_full = np.full((T_valid, *config.IMG_SHAPE), np.nan, dtype=np.float32)
    cov_img_full = np.zeros((T_valid, F_dim, *config.IMG_SHAPE), dtype=np.float32) 
    
    logging.info(f"⏳ 正在将一维序列映射为 32x32 空间图...")
    for t_idx, original_t in enumerate(valid_time_indices):
        # 填充稀疏浓度图 (仅有站点的地方有值)
        for grid_idx in station_grid_idx:
            row, col = grid_idx // config.IMG_SHAPE[1], grid_idx % config.IMG_SHAPE[1]
            conc_img_full[t_idx, row, col] = conc_raw_norm[original_t, grid_idx]
            
        # 填充多源特征图 (假设全部 999 个网格都有协变量数据，形成稠密底图)
        for grid_idx in range(config.NUM_GRIDS):
            row, col = grid_idx // config.IMG_SHAPE[1], grid_idx % config.IMG_SHAPE[1]
            cov_img_full[t_idx, :, row, col] = features_norm[original_t, grid_idx, :]

    # 7. 保存数据集
    time_splits = {
        "train": np.arange(len(valid_train_indices)),
        "val": np.arange(len(valid_train_indices), len(valid_train_indices) + len(valid_val_indices)),
        "test": np.arange(len(valid_train_indices) + len(valid_val_indices), T_valid)
    }
    
    save_dir = os.path.join(config.OUTPUT_DIR, "data", pollutant)
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "conc_img_full.npy"), conc_img_full)
    np.save(os.path.join(save_dir, "cov_img_full.npy"), cov_img_full) # 核心！保存多源外挂特征图
    np.save(os.path.join(save_dir, "coord_img.npy"), coord_img)
    np.save(os.path.join(save_dir, "time_splits.npy"), time_splits)
    np.save(os.path.join(save_dir, "valid_time_indices.npy"), valid_time_indices)
    np.save(os.path.join(save_dir, "fixed_global_stats.npy"), np.array([fixed_global_mean, fixed_global_std]))
    joblib.dump(conc_scaler, os.path.join(save_dir, "conc_scaler.pkl"))
    joblib.dump(feat_scaler, os.path.join(save_dir, "feat_scaler.pkl"))
    np.save(os.path.join(save_dir, "station_positions.npy"), station_positions)
    np.save(os.path.join(save_dir, "station_grid_idx.npy"), station_grid_idx)
    
    # 画一张示例验证图
    vis_dir = os.path.join(config.OUTPUT_DIR, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.heatmap(conc_img_full[0], cmap="viridis", cbar=True, ax=ax)
    ax.scatter(station_positions[:,1]+0.5, station_positions[:,0]+0.5, color='red', s=15, label='Stations')
    ax.set_title(f"{pollutant.upper()} Concentration (T=0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{pollutant}_station_vis.png"), bbox_inches="tight")
    plt.close()
    
    logging.info(f"🎉 {pollutant.upper()} 预处理及多源特征映射完成！")
    
    del conc_raw_data, features_raw_data, conc_raw_norm, conc_img_full, cov_img_full
    gc.collect()

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 修复多进程启动
    mp.set_start_method('spawn', force=True)
    # 调高系统文件句柄上限
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, hard))
    logging.info(f"✅ 系统文件句柄限制已调高至：{soft} → 65535")
    
    config = Config()
    try:
        check_all_paths(config)
        station_indices, station_positions, station_grid_idx, global_mask, non_station_grid_idx = load_station_info(config)
        
        # 1. 处理核心污染物数据，生成基于多源特征的基准的时间对齐规则
        # 1. 处理核心污染物数据，生成基于多源特征的基准的时间对齐规则
        for pollutant in config.POLLUTANTS:
            try:
                # ⬇️ 删掉了 non_station_grid_idx，现在是 5 个参数，完美匹配
                process_pollutant_data_loocv(config, pollutant, station_grid_idx, global_mask, station_positions)
            except Exception as e:
                logging.error(f"❌ {pollutant.upper()} 处理失败，跳过：{str(e)}")
                continue
                
        logging.info(f"\n✅ 所有污染物预处理完成！数据保存到：{config.OUTPUT_DIR}/data")
        logging.info(f"👉 预处理全部搞定！接下来运行 rh.py 进行留一法验证与全区域预测")
    
    except Exception as e:
        logging.error(f"主流程执行失败：{str(e)}")
        raise