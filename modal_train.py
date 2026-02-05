import modal
import os
import subprocess

ignore_patterns = [
    ".venv", ".idea", "__pycache__", "data", ".idea", "processed.tar.gz", "venv"
]
# 1. Khởi tạo App
app = modal.App("transattunet-lidc-training")

# 2. Cấu hình môi trường chạy
REMOTE_ROOT = "/root/project"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(".", remote_path=REMOTE_ROOT, copy=True,
                   ignore=ignore_patterns)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pyyaml", "tqdm", "scikit-learn", "scipy")
)

volume = modal.Volume.from_name("storage", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/mnt/storage": volume},
    timeout=86400
)
def train_remote(resume_path: str = None):
    import subprocess
    import os
    import sys

    # Thiết lập môi trường
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT
    os.chdir(REMOTE_ROOT)

    print(" Đang khởi động Training trên Modal GPU...")

    # Xây dựng lệnh chạy
    cmd = [
        "python", "train.py",
        "--config", "configs/config.yaml"
    ]

    # Nếu có tham số resume, thêm vào lệnh chạy
    if resume_path:
        print(f"Khôi phục huấn luyện từ checkpoint: {resume_path}")
        cmd.extend(["--resume", resume_path])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

# ==================== INFERENCE / EVALUATION ====================
@app.function(
    image=image,
    gpu="A100",
    volumes={"/mnt/storage": volume},  # Volume được gắn ở /mnt/storage
    timeout=7200,
)
def evaluate_remote(
        model_path: str,
        vis_num: int = 50,
        config_path: str = "configs/config.yaml"
):
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT
    os.chdir(REMOTE_ROOT)

    # Đường dẫn lưu trên Volume (Ổ cứng bền vững)
    volume_output_dir = "/mnt/storage/results_roi"
    os.makedirs(volume_output_dir, exist_ok=True)

    print(f"Đang chạy Evaluation... Kết quả sẽ lưu tại: {volume_output_dir}")

    cmd = [
        "python", "inference.py",
        "--config", config_path,
        "--model_path", model_path,
        "--vis_num", str(vis_num),
        "--save_dir", volume_output_dir  # <--- TRUYỀN ĐƯỜNG DẪN VOLUME VÀO ĐÂY
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="")
    process.wait()

    print("\n✅ Đã chạy xong! Dữ liệu đã an toàn trên Volume.")

# 3. Hàm main tại Local để nhận tham số dòng lệnh
@app.local_entrypoint()
def main(resume: str = None):
    """
    Cách chạy:
    1. Train mới: modal run -d modal_train.py::main
    2. Resume:    modal run -d modal_train.py::main --resume /mnt/storage/outputs/checkpoints/last_checkpoint.pth
    """
    """Load volume
    tạo volume 
    modal volume create storage
    # Upload folder data đã xử lý lên Volume
    modal volume put storage "D:\Workspace\TransAttUnet-LIDC\data\processed" /data/processed
    """
    train_remote.remote(resume_path=resume)

# ==================== LOCAL ENTRYPOINTS ====================
@app.local_entrypoint()
def train(resume: str = None):
    """Chạy training"""
    train_remote.remote(resume_path=resume)


@app.local_entrypoint()
def evaluate(
    model_path: str = modal.parameter(),
    vis_num: int = 50
):
    """
    Chạy evaluation/inference trên tập test.

    Ví dụ sử dụng:
    modal run modal_train.py::evaluate --model-path /mnt/storage/outputs/checkpoints/best_model.pth --vis-num 100
    """
    evaluate_remote.remote(
        model_path=model_path,
        vis_num=vis_num,
        config_path="configs/config.yaml"
    )