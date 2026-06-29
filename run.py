"""
run.py — 逆向识别算法测试脚本

用法：
    python run.py

依赖：确保已安装 requirements.txt 中的所有依赖。
"""
import os
import torch
from inverse_model.inverse_config import InverseConfig
from inverse_model.inverse_architecture import InverseOperator
from inverse_model.inverse_dataset import InverseWindowDataset, WindowConfig
from inverse_model.inverse_trainer import InverseTrainer

RESULTS_PATH = r"results/高速客车-外部导入-20260317_234815/files/simulation_result.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 配置加载
cfg = InverseConfig()
win_cfg = WindowConfig(window_size=256, stride=256)

# 2. 数据集加载（单文件推理）
dataset = InverseWindowDataset(RESULTS_PATH, cfg, win_cfg, split="test", val_ratio=0.0)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# 3. 模型初始化
model = InverseOperator(cfg).to(DEVICE)
model.eval()

# 4. 可选：加载已训练权重（如有 best_model.pt）
ckpt_path = "best_model.pt"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    print(f"Loaded checkpoint from {ckpt_path}")
else:
    print("未检测到 best_model.pt，使用随机初始化模型，仅做结构测试。")

# 5. 推理与结果保存
os.makedirs("test_outputs", exist_ok=True)

for i, batch in enumerate(loader):
    y = batch["y"].to(DEVICE).float()           # [1, T, n_sensors]
    xq = batch["x_query"].to(DEVICE).float()    # [1, L]
    c = batch["c"].to(DEVICE).float()           # [1, n_cond]
    with torch.no_grad():
        z_pred = model(y, xq, c)                # [1, L, n_dir]
    # 保存结果
    out_path = f"test_outputs/inverse_pred_{i:03d}.npy"
    np_z_pred = z_pred.cpu().numpy().squeeze()
    import numpy as np
    np.save(out_path, np_z_pred)
    print(f"Saved: {out_path}")

print("推理完成。结果已保存于 test_outputs/ 文件夹下。")
