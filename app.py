from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time
from datetime import datetime
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import traceback
from contextlib import asynccontextmanager
import psutil  # For CPU usage monitoring

# --- PyNVML Initialization for GPU Monitoring ---
PYNVML_AVAILABLE = False
PYNVML_INITIALIZED_SUCCESSFULLY = False
try:
    import pynvml
    PYNVML_AVAILABLE = True
    # print("PyNVML library found. GPU monitoring will be attempted.")  # Commented out to avoid startup noise
except ImportError:
    # print("PyNVML (nvidia-ml-py) library not found. GPU monitoring will be disabled. Install with 'pip install nvidia-ml-py3'")  # Commented out
    pynvml = None
except Exception as e:
    print(f"Unexpected error during PyNVML import: {e}")
    pynvml = None

JOBS_DIR = "training_jobs"
CONFIGS_DIR = "model_configs"
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

active_jobs: Dict[str, Dict[str, Any]] = {}
model_configs: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PYNVML_INITIALIZED_SUCCESSFULLY
    # Startup logic
    if PYNVML_AVAILABLE and pynvml:
        try:
            pynvml.nvmlInit()
            PYNVML_INITIALIZED_SUCCESSFULLY = True
            print("PyNVML initialized globally for GPU monitoring at application startup.")
        except pynvml.NVMLError as e:
            print(f"Global PyNVML Initialization Error: {e}. GPU monitoring might be unreliable.")
            PYNVML_INITIALIZED_SUCCESSFULLY = False
        except Exception as e:
            print(f"Unexpected error during global PyNVML Initialization: {e}")
            PYNVML_INITIALIZED_SUCCESSFULLY = False

    # Load persisted job statuses into active_jobs
    print("Loading persisted job statuses at startup...")
    for job_file_name in os.listdir(JOBS_DIR):
        if job_file_name.endswith(".json"):
            job_id = job_file_name.replace(".json", "")
            status_data = load_job_status(job_id)
            if status_data:
                active_jobs[job_id] = status_data
    print(f"Loaded {len(active_jobs)} job statuses into memory.")

    # Load persisted model configurations
    print("Loading persisted model configurations at startup...")
    for config_file_name in os.listdir(CONFIGS_DIR):
        if config_file_name.endswith(".json"):
            config_id = config_file_name.replace(".json", "")
            config_data = load_model_config(config_id)
            if config_data:
                model_configs[config_id] = config_data
    print(f"Loaded {len(model_configs)} model configurations into memory.")

    yield  # This is where your application will run

    # Shutdown logic
    if PYNVML_INITIALIZED_SUCCESSFULLY and pynvml:
        try:
            pynvml.nvmlShutdown()
            print("PyNVML shut down globally.")
        except pynvml.NVMLError as e:
            print(f"Error during global PyNVML shutdown: {e}")
        except Exception as e:
            print(f"Unexpected error during global PyNVML shutdown: {e}")

app = FastAPI(title="ZPE Quantum Neural Network Training API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingParameters(BaseModel):
    totalEpochs: int
    batchSize: int
    learningRate: float
    weightDecay: float
    momentumParams: List[float]
    strengthParams: List[float]
    noiseParams: List[float]
    couplingParams: Optional[List[float]] = None  # Added couplingParams as optional
    quantumCircuitSize: int
    labelSmoothing: float
    quantumMode: bool
    modelName: str
    baseConfigId: Optional[str] = None

class ModelConfig(BaseModel):
    id: Optional[str] = None
    name: str
    parameters: TrainingParameters
    date_created: str
    accuracy: float
    loss: float
    use_quantum_noise: bool
    channel_sizes: Optional[List[int]] = None  # Added channel_sizes as optional

class ZPEDeepNet(nn.Module):
    def __init__(self, input_channels=1, output_size=10, sequence_length=10):
        super(ZPEDeepNet, self).__init__()
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zpe_flows = [torch.ones(sequence_length, device=self.device) for _ in range(6)]
        self.cycle_length = 2 ** 5
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )
        fc_input_features = 512 * 1 * 1
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_features, 2048), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )
        self.shortcut1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(2))
        self.shortcut2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(2))
        self.shortcut3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(2))
        self.shortcut4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(2))

    def perturb_zpe_flow(self, data, zpe_idx, feature_size):
        if data.numel() == 0: return
        batch_mean = torch.mean(data.detach(), dim=0).view(-1)
        if batch_mean.numel() == 0 or self.sequence_length == 0: return
        divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
        if divisible_size == 0:
            if batch_mean.size(0) > 0:
                padding_size = self.sequence_length - batch_mean.size(0)
                if padding_size < 0: padding_size = 0
                batch_mean_padded = torch.cat((batch_mean, torch.zeros(padding_size, device=self.device)))
                reshaped = batch_mean_padded.view(-1, self.sequence_length)
            else: return
        else:
            batch_mean_truncated = batch_mean[:divisible_size]
            reshaped = batch_mean_truncated.view(-1, self.sequence_length)
        if reshaped.numel() == 0: return
        perturbation = torch.mean(reshaped, dim=0)
        perturbation = torch.tanh(perturbation * 0.3)
        momentum = 0.9 if zpe_idx < 4 else 0.7
        with torch.no_grad():
            self.zpe_flows[zpe_idx].data = momentum * self.zpe_flows[zpe_idx].data + (1 - momentum) * (1.0 + perturbation)
            self.zpe_flows[zpe_idx].data = torch.clamp(self.zpe_flows[zpe_idx].data, 0.8, 1.2)

    def apply_zpe(self, x, zpe_idx, spatial=True):
        if x.numel() == 0: return x
        feature_size = x.size(1) if spatial else x.size(-1)
        if feature_size == 0 and x.numel() > 0:
            if not spatial and x.dim() == 2: feature_size = x.size(-1)
            else: return x
        elif feature_size == 0 and x.numel() == 0: return x
        if self.sequence_length == 0: return x

        # Perturb the ZPE flow based on data
        self.perturb_zpe_flow(x, zpe_idx, feature_size)

        # Apply a basic classical approximation of the "000000.1" concept
        # Dynamically adjust the flow based on its bit representation and cycle_length
        flow = self.zpe_flows[zpe_idx]
        if spatial:
            if x.size(2) == 0 or x.size(3) == 0: return x
            num_elements_to_cover = x.size(2) * x.size(3)
            repeats = (num_elements_to_cover + self.sequence_length - 1) // self.sequence_length
            flow_expanded_flat = flow.repeat(repeats)[:num_elements_to_cover]
            flow_expanded = flow_expanded_flat.view(1, 1, x.size(2), x.size(3))
            if x.size(1) > 0: flow_expanded = flow_expanded.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        else:
            if x.size(-1) == 0: return x
            num_elements_to_cover = x.size(-1)
            repeats = (num_elements_to_cover + self.sequence_length - 1) // self.sequence_length
            flow_expanded_flat = flow.repeat(repeats)[:num_elements_to_cover]
            flow_expanded = flow_expanded_flat.view(1, -1).expand_as(x)
        return x * flow_expanded

    def forward(self, x):
        x = self.apply_zpe(x, 0); residual = self.shortcut1(x); x = self.conv1(x) + residual
        x = self.apply_zpe(x, 1); residual = self.shortcut2(x); x = self.conv2(x) + residual
        x = self.apply_zpe(x, 2); residual = self.shortcut3(x); x = self.conv3(x) + residual
        x = self.apply_zpe(x, 3); residual = self.shortcut4(x); x = self.conv4(x) + residual
        x = self.apply_zpe(x, 4); x = self.fc(x)
        x = self.apply_zpe(x, 5, spatial=False)
        return x

    def analyze_zpe_effect(self):
        return [torch.mean(torch.abs(flow - 1.0)).item() for flow in self.zpe_flows]

def mixup(data, targets, alpha=1.0, device='cpu'):
    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data, shuffled_targets = data[indices], targets[indices]
    lam = np.random.beta(alpha, alpha)
    return lam * data + (1 - lam) * shuffled_data, targets, shuffled_targets, lam

def get_gpu_usage_info_internal() -> Dict[str, Any]:
    if not PYNVML_AVAILABLE or not pynvml:
        return {"error": "PyNVML library not available/imported. Cannot fetch GPU stats."}
    if not PYNVML_INITIALIZED_SUCCESSFULLY:
        return {"error": "PyNVML not initialized successfully at startup. Cannot fetch GPU stats."}
    try:
        num_gpus = pynvml.nvmlDeviceGetCount()
        if num_gpus == 0: return {"info": "No NVIDIA GPUs detected."}
        gpu_info_list = []
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power_draw_mw = None
            try: power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            except pynvml.NVMLError_NotSupported: power_draw_mw = None
            fan_speed = None
            try: fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)  # For single fan
            except pynvml.NVMLError_NotSupported: fan_speed = None
            except Exception: fan_speed = None  # Catch other errors like if device has 0 fans
            gpu_name_bytes = pynvml.nvmlDeviceGetName(handle)
            gpu_name = gpu_name_bytes.decode('utf-8') if isinstance(gpu_name_bytes, bytes) else gpu_name_bytes
            gpu_info_list.append({
                "id": str(i), "name": gpu_name,
                "utilization_gpu_percent": float(util.gpu),
                "utilization_memory_io_percent": float(util.memory),
                "memory_total_mb": float(mem_info.total / (1024**2)),
                "memory_used_mb": float(mem_info.used / (1024**2)),
                "memory_free_mb": float(mem_info.free / (1024**2)),
                "memory_used_percent": float(mem_info.used * 100 / mem_info.total) if mem_info.total > 0 else 0.0,
                "temperature_c": float(temp),
                "power_draw_w": float(power_draw_mw / 1000.0) if power_draw_mw is not None else None,
                "fan_speed_percent": float(fan_speed) if fan_speed is not None else None,
            })
        return gpu_info_list[0] if gpu_info_list else {"info": "No GPU stats could be retrieved."}
    except pynvml.NVMLError as e:
        return {"error": f"NVML Error fetching GPU stats: {str(e)}"}
    except Exception as e:
        return {"error": f"General Error fetching GPU stats: {str(e)}"}

def get_cpu_usage_info_detailed() -> Dict[str, Any]:
    """Fetches detailed CPU usage and frequency."""
    try:
        # Get overall CPU usage percentage
        overall_usage = psutil.cpu_percent(interval=0.1)
        # Get current CPU frequency
        frequency = psutil.cpu_freq()
        return {
            "overall_usage_percent": float(overall_usage),
            "current_frequency_mhz": float(frequency.current) if frequency else None,
        }
    except Exception as e:
        return {"error": f"Error fetching CPU stats: {str(e)}"}

def save_job_status(job_id: str):
    if job_id not in active_jobs: return
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    try:
        with open(job_file, "w") as f: json.dump(active_jobs[job_id], f, indent=2)
    except Exception as e: print(f"Error saving job status for {job_id}: {e}")

def load_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if os.path.exists(job_file):
        try:
            with open(job_file, "r") as f: return json.load(f)
        except Exception as e: print(f"Error loading job status for {job_id}: {e}")
    return None

# Functions for saving and loading model configurations
def save_model_config(config_id: str):
    if config_id not in model_configs: return
    config_file = os.path.join(CONFIGS_DIR, f"{config_id}.json")
    try:
        with open(config_file, "w") as f: json.dump(model_configs[config_id], f, indent=2)
    except Exception as e: print(f"Error saving model config for {config_id}: {e}")

def load_model_config(config_id: str) -> Optional[Dict[str, Any]]:
    config_file = os.path.join(CONFIGS_DIR, f"{config_id}.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f: return json.load(f)
        except Exception as e: print(f"Error loading model config for {config_id}: {e}")
    return None

def run_training_job(job_id: str, params: TrainingParameters):
    global active_jobs
    if job_id not in active_jobs:
        loaded_status = load_job_status(job_id)
        if loaded_status: active_jobs[job_id] = loaded_status
        else:
            active_jobs[job_id] = {
                "job_id": job_id, "status": "failed", "log_messages": [f"Job ID {job_id} not found."],
                "parameters": params.model_dump(), "total_epochs": params.totalEpochs, "current_epoch": 0,
                "accuracy": 0.0, "loss": 0.0, "zpe_effects": [], "zpe_history": [],
                "start_time": datetime.now().isoformat(), "end_time": datetime.now().isoformat(),
                "gpu_info": get_gpu_usage_info_internal()
            }
            save_job_status(job_id)
            return

    job_status = active_jobs[job_id]
    job_status["status"] = "running"
    job_status["log_messages"].append(f"Starting PyTorch training: {params.modelName}")
    job_status["start_time"] = datetime.now().isoformat()
    job_status["gpu_info"] = get_gpu_usage_info_internal()
    save_job_status(job_id)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        job_status["log_messages"].append(f"Device: {device}")
        job_status["log_messages"].append("Setting up MNIST dataset...")
        train_transform = transforms.Compose([
            transforms.RandomRotation(15), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomCrop(28, padding=2), transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        ])
        val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset_full = datasets.MNIST(root='./data_api', train=True, download=True, transform=train_transform)
        val_dataset = datasets.MNIST(root='./data_api', train=False, download=True, transform=val_transform)
        train_loader = DataLoader(train_dataset_full, batch_size=params.batchSize, shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(val_dataset, batch_size=params.batchSize, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))
        job_status["log_messages"].append(f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)} batches.")
        model = ZPEDeepNet(input_channels=1, output_size=10, sequence_length=10).to(device)  # MNIST specific
        criterion = nn.CrossEntropyLoss(label_smoothing=params.labelSmoothing)
        optimizer = optim.Adam(model.parameters(), lr=params.learningRate, weight_decay=params.weightDecay)
        scheduler = CosineAnnealingLR(optimizer, T_max=params.totalEpochs)
        job_status["log_messages"].append("Model & Optimizer initialized.")
        job_status["gpu_info"] = get_gpu_usage_info_internal()
        save_job_status(job_id)

        for epoch in range(1, params.totalEpochs + 1):
            if job_status["status"] == "stopped": job_status["log_messages"].append("Training stopped."); break
            model.train(); epoch_loss_sum, num_batches_epoch = 0.0, 0
            job_status["log_messages"].append(f"--- Epoch {epoch}/{params.totalEpochs} ---")
            job_status["gpu_info"] = get_gpu_usage_info_internal()
            save_job_status(job_id)

            for batch_idx, (data, target) in enumerate(train_loader):
                if job_status["status"] == "stopped": break
                data, target = data.to(device), target.to(device)
                data_mixed, target_a, target_b, lam = mixup(data, target, alpha=0.4, device=device)
                optimizer.zero_grad(); output = model(data_mixed)
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                loss.backward(); optimizer.step()
                epoch_loss_sum += loss.item(); num_batches_epoch += 1
                if batch_idx > 0 and batch_idx % (len(train_loader) // 5 or 1) == 0:
                    job_status["log_messages"].append(f"E{epoch} B{batch_idx}/{len(train_loader)} L: {loss.item():.4f}")
                    job_status.update({
                        "current_epoch": epoch, "loss": loss.item(),
                        "zpe_effects": model.analyze_zpe_effect(),
                        "gpu_info": get_gpu_usage_info_internal(),
            })
                    try: save_job_status(job_id)
                    except Exception as e: print(f"Error saving job status during epoch {epoch}, batch {batch_idx}: {e}")

            scheduler.step()
            avg_epoch_train_loss = epoch_loss_sum / num_batches_epoch if num_batches_epoch > 0 else 0.0
            model.eval(); val_correct, val_total, val_loss_sum, val_batches_epoch = 0, 0, 0.0, 0
            with torch.no_grad():
                for data_val, target_val in val_loader:
                    data_val, target_val = data_val.to(device), target_val.to(device)
                    output_val = model(data_val); val_loss = criterion(output_val, target_val)
                    val_loss_sum += val_loss.item(); val_batches_epoch += 1
                    _, predicted_val = torch.max(output_val.data, 1)
                    val_total += target_val.size(0); val_correct += (predicted_val == target_val).sum().item()
            val_accuracy = (100 * val_correct / val_total) if val_total > 0 else 0.0
            avg_epoch_val_loss = val_loss_sum / val_batches_epoch if val_batches_epoch > 0 else 0.0
            job_status.update({
                "current_epoch": epoch, "accuracy": val_accuracy, "loss": avg_epoch_val_loss,
                "zpe_effects": model.analyze_zpe_effect(),
                "gpu_info": get_gpu_usage_info_internal()
            })
            # Ensure zpe_history is a list and append current epoch's ZPE effects
            if 'zpe_history' in job_status and isinstance(job_status['zpe_history'], list):
 job_status["zpe_history"].append({
 "epoch": epoch,
 "zpe_effects": job_status["zpe_effects"]
            })
            job_status["log_messages"].append(f"E{epoch} END - TrainL: {avg_epoch_train_loss:.4f}, ValAcc: {val_accuracy:.2f}%, ValL: {avg_epoch_val_loss:.4f}")
            job_status["log_messages"].append(f"ZPE: {[float(f'{x:.4f}') for x in job_status['zpe_effects']]}")
            save_job_status(job_id)

        if job_status["status"] != "stopped":
            job_status["status"] = "completed"
            final_msg = f"Training done! Final Val Acc: {job_status['accuracy']:.2f}%"
            job_status["log_messages"].append(final_msg); print(f"Job {job_id}: {final_msg}")
            model_save_path = os.path.join(JOBS_DIR, f"{job_id}_model.pth")
            torch.save(model.state_dict(), model_save_path)
            job_status["log_messages"].append(f"Model saved: {model_save_path}")
    except Exception as e:
        job_status["status"] = "failed"; error_msg = f"Train Error job {job_id}: {str(e)}"
        tb_str = traceback.format_exc()
        job_status["log_messages"].extend([error_msg, tb_str]); print(error_msg, "\n", tb_str)
    finally:
        job_status["end_time"] = datetime.now().isoformat()
        job_status["gpu_info"] = get_gpu_usage_info_internal()
        save_job_status(job_id)

@app.post("/api/train")
async def start_training_endpoint(params: TrainingParameters, background_tasks: BackgroundTasks):
    job_id = f"zpe_job_{str(uuid.uuid4())[:8]}"
    active_jobs[job_id] = {
        "job_id": job_id, "status": "pending", "current_epoch": 0, "zpe_history": [], # Initialize zpe_history
        "total_epochs": params.totalEpochs, "accuracy": 0.0, "loss": 0.0,
        "zpe_effects": [0.0] * 6,
        "log_messages": [f"Init job: {params.modelName}"],
        "parameters": params.model_dump(), "start_time": None, "end_time": None,
        "gpu_info": get_gpu_usage_info_internal()
    }
    save_job_status(job_id)
    background_tasks.add_task(run_training_job, job_id, params)
    return {"status": "training_started", "job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_job_status_endpoint(job_id: str):
    if job_id in active_jobs and active_jobs[job_id]["status"] in ["running", "pending"]:
        if active_jobs[job_id]["status"] == "running": active_jobs[job_id]["gpu_info"] = get_gpu_usage_info_internal()
        return active_jobs[job_id]
    job_status_from_file = load_job_status(job_id)
    if job_status_from_file:
        if job_status_from_file["status"] in ["completed", "failed", "stopped"]:
            active_jobs[job_id] = job_status_from_file
            # No need to save again here unless the GPU info was updated, which we are only doing for running jobs
        return job_status_from_file
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

@app.post("/api/stop/{job_id}")
async def stop_training_endpoint(job_id: str):
    if job_id in active_jobs and active_jobs[job_id]["status"] in ["running", "pending"]:
        active_jobs[job_id]["status"] = "stopped"
        active_jobs[job_id]["log_messages"].append("Stop request received.")
        save_job_status(job_id)
        return {"status": "stop_requested", "job_id": job_id}
    job_status_from_file = load_job_status(job_id)
    if job_status_from_file: return {"status": job_status_from_file["status"], "message": "Job not active.", "job_id": job_id}
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

@app.get("/api/jobs")
async def list_jobs_endpoint(limit: int = 20):
    processed_job_ids, jobs_summary_list = set(), []
    # Prioritize active_jobs
    for job_id, status_dict in list(active_jobs.items()):
        model_name = status_dict.get("parameters", {}).get("modelName", "Unknown")
        jobs_summary_list.append({
            "job_id": job_id, "model_name": model_name,
            "status": status_dict.get("status", "unknown"), "accuracy": status_dict.get("accuracy", 0.0),
            "current_epoch": status_dict.get("current_epoch", 0), "total_epochs": status_dict.get("total_epochs", 0),
            "start_time": status_dict.get("start_time")
        }); processed_job_ids.add(job_id)
    try:
        job_files = sorted(
            [f for f in os.listdir(JOBS_DIR) if f.endswith('.json')],
            key=lambda f: os.path.getmtime(os.path.join(JOBS_DIR, f)), reverse=True
        )
        for job_file_name in job_files:
            # Only load from disk if we haven't reached the limit and the job isn't already in active_jobs
            if len(jobs_summary_list) < limit + len(active_jobs) and job_file_name.replace(".json", "") not in processed_job_ids:
                job_id_from_file = job_file_name.replace(".json", "")
                job_data = load_job_status(job_id_from_file)
                if job_data:
                    model_name_file = job_data.get("parameters", {}).get("modelName", "Unknown")
                    jobs_summary_list.append({
                        "job_id": job_id_from_file, "model_name": model_name_file,
                        "status": job_data.get("status", "unknown"), "accuracy": job_data.get("accuracy", 0.0),
                        "current_epoch": job_data.get("current_epoch", 0), "total_epochs": job_data.get("total_epochs", 0),
                        "start_time": job_data.get("start_time")
                    })
    except Exception as e: print(f"Error listing jobs from disk: {e}")

    # Deduplicate based on job_id (active jobs were added first, so they take precedence)
    # Then sort and limit
    final_jobs_map = {job['job_id']: job for job in jobs_summary_list}
    sorted_jobs = sorted(final_jobs_map.values(), key=lambda x: x.get("start_time", ""), reverse=True)

    return {"jobs": sorted_jobs[:limit]}

@app.get("/api/gpu-stats")
async def get_system_gpu_stats_endpoint():
    return get_gpu_usage_info_internal()

@app.get("/api/system-stats")
async def get_system_general_stats_endpoint():
    gpu_data = get_gpu_usage_info_internal()
    primary_gpu_info = None
    if isinstance(gpu_data, dict) and ("error" in gpu_data or "info" in gpu_data or "id" in gpu_data):
        primary_gpu_info = gpu_data  # Handles single GPU, error, or info dict
    # Note: get_gpu_usage_info_internal() already returns primary GPU or error/info if no list

    return {
        "gpu_info": primary_gpu_info,
        "cpu_info": get_cpu_usage_info_detailed()
    }

# Endpoint to save model configurations
@app.post("/api/configs")
async def create_model_config_endpoint(config: ModelConfig):
    config_id = f"config_{str(uuid.uuid4())[:8]}"
    config.id = config_id  # Assign a new ID to the incoming config
    model_configs[config_id] = config.model_dump()  # Store the config
    save_model_config(config_id)  # Persist the config to disk
    return {"status": "configuration_saved", "config_id": config_id}

# Endpoint to get a single model configuration by ID
@app.get("/api/configs/{config_id}")
async def get_model_config_endpoint(config_id: str):
    if config_id in model_configs:
        return model_configs[config_id]
    config_data_from_file = load_model_config(config_id)
    if config_data_from_file:
        model_configs[config_id] = config_data_from_file  # Load into memory if found on disk
        return config_data_from_file
    raise HTTPException(status_code=404, detail=f"Model configuration {config_id} not found")

# Endpoint to list all model configurations
@app.get("/api/configs")
async def list_model_configs_endpoint(limit: int = 20):
    configs_list = list(model_configs.values())  # Get configs currently in memory

    # Also load any configs from disk that aren't in memory yet
    try:
        config_files = os.listdir(CONFIGS_DIR)
        for config_file_name in config_files:
            if config_file_name.endswith('.json'):
                config_id_from_file = config_file_name.replace(".json", "")
                if config_id_from_file not in model_configs:
                    config_data = load_model_config(config_id_from_file)
                    if config_data:
                        model_configs[config_id_from_file] = config_data  # Load into memory
                        configs_list.append(config_data)  # Add to the list
    except Exception as e: print(f"Error listing model configurations from disk: {e}")

    return {"configs": configs_list}

@app.delete("/api/configs/{config_id}")
async def delete_model_config_endpoint(config_id: str):
    # Remove from in-memory dictionary
    if config_id in model_configs:
        del model_configs[config_id]

    # Remove from disk
    config_file_path = os.path.join(CONFIGS_DIR, f"{config_id}.json")
    if os.path.exists(config_file_path):
        try:
            os.remove(config_file_path)
        except Exception as e:
            print(f"Error deleting config file {config_file_path}: {e}")
            # Potentially re-add to memory if consistency is paramount and disk deletion fails
            # For now, assume in-memory success is primary
        return {"status": "configuration_deleted", "config_id": config_id}

    # If not in memory and not on disk, it's effectively not found
    if config_id not in model_configs and not os.path.exists(config_file_path):
        raise HTTPException(status_code=404, detail=f"Model configuration {config_id} not found to delete.")

    return {"status": "configuration_deleted_from_memory_only", "config_id": config_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)