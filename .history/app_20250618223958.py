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
import pandas as pd  # For dataset preparation
# Ensure you have a module or file named your_ai_flows_module.py
# containing an async function advise_hsqnn_parameters_flow
# from your_ai_flows_module import advise_hsqnn_parameters_flow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging
from src.lib.training_logger import TrainingLogger
from mixup_alpha_advisor import router as mixup_alpha_router

# GPU Monitoring Setup
PYNVML_AVAILABLE = False
PYNVML_INITIALIZED_SUCCESSFULLY = False
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
except Exception as e:
    print(f"Unexpected error during PyNVML import: {e}")
    pynvml = None

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories for Jobs, Configs, and Datasets
JOBS_DIR = "training_jobs"
CONFIGS_DIR = "model_configs"
DATASET_DIR = "datasets"
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# In-Memory Storage
active_jobs: Dict[str, Dict[str, Any]] = {}
model_configs: Dict[str, Dict[str, Any]] = {}

# Initialize the training logger
training_logger = TrainingLogger()

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global PYNVML_INITIALIZED_SUCCESSFULLY
    # Startup Logic
    if PYNVML_AVAILABLE and pynvml:
        try:
            pynvml.nvmlInit()
            PYNVML_INITIALIZED_SUCCESSFULLY = True
            logger.info("PyNVML initialized for GPU monitoring.")
        except pynvml.NVMLError as e:
            logger.error(f"PyNVML Initialization Error: {e}")
            PYNVML_INITIALIZED_SUCCESSFULLY = False
        except Exception as e:
            logger.error(f"Unexpected error during PyNVML Initialization: {e}")
            PYNVML_INITIALIZED_SUCCESSFULLY = False

    # Load Persisted Jobs
    logger.info("Loading persisted job statuses...")
    for job_file_name in os.listdir(JOBS_DIR):
        if job_file_name.endswith(".json"):
            job_id = job_file_name.replace(".json", "")
            status_data = load_job_status(job_id)
            if status_data:
                active_jobs[job_id] = status_data
    logger.info(f"Loaded {len(active_jobs)} job statuses.")

    # Load Persisted Configs
    logger.info("Loading persisted model configurations...")
    for config_file_name in os.listdir(CONFIGS_DIR):
        if config_file_name.endswith(".json"):
            config_id = config_file_name.replace(".json", "")
            config_data = load_model_config(config_id)
            if config_data:
                model_configs[config_id] = config_data
    logger.info(f"Loaded {len(model_configs)} model configurations.")

    yield  # Application Runs Here

    # Shutdown Logic
    if PYNVML_INITIALIZED_SUCCESSFULLY and pynvml:
        try:
            pynvml.nvmlShutdown()
            logger.info("PyNVML shut down.")
        except pynvml.NVMLError as e:
            logger.error(f"PyNVML shutdown error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PyNVML shutdown: {e}")

# FastAPI Application
app = FastAPI(
    title="ZPE Quantum Neural Network Training API",
    description="API for training quantum-enhanced PyTorch models and preparing datasets.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class TrainingParameters(BaseModel):
    totalEpochs: int
    batchSize: int
    learningRate: float
    weightDecay: float
    momentumParams: List[float]
    strengthParams: List[float]
    noiseParams: List[float]
    couplingParams: Optional[List[float]] = None
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
    channel_sizes: Optional[List[int]] = None

class ChatMessage(BaseModel):
    message: str
    job_id: Optional[str] = None  # Added to allow explicit job ID specification

class DatasetPrepRequest(BaseModel):
    input_file: str = "pytorch_quantum_dataset.csv"
    target_column: Optional[str] = None
    feature_columns: List[str] = ["content", "description"]
    output_file: str = "processed_dataset.csv"

# ZPEDeepNet Model
class ZPEDeepNet(nn.Module):
    def __init__(self, input_channels=1, output_size=10, sequence_length=10):
        super(ZPEDeepNet, self).__init__()
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zpe_flows = [torch.ones(sequence_length, device=self.device) for _ in range(6)]
        self.cycle_length = 2 ** 5
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        fc_input_features = 512 * 1 * 1
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )
        self.shortcut1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(2))
        self.shortcut2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(2))
        self.shortcut3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(2))
        self.shortcut4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(2))

    def perturb_zpe_flow(self, data, zpe_idx, feature_size):
        if data.numel() == 0:
            return
        batch_mean = torch.mean(data.detach(), dim=0).reshape(-1)
        if batch_mean.numel() == 0 or self.sequence_length == 0:
            return
        divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
        if divisible_size == 0:
            if batch_mean.size(0) > 0:
                padding_size = self.sequence_length - batch_mean.size(0)
                if padding_size < 0:
                    padding_size = 0
                batch_mean_padded = torch.cat((batch_mean, torch.zeros(padding_size, device=self.device)))
                reshaped = batch_mean_padded.reshape(-1, self.sequence_length)
            else:
                return
        else:
            batch_mean_truncated = batch_mean[:divisible_size]
            reshaped = batch_mean_truncated.reshape(-1, self.sequence_length)
        if reshaped.numel() == 0:
            return
        perturbation = torch.mean(reshaped, dim=0)
        perturbation = torch.tanh(perturbation * 0.3)
        momentum = 0.9 if zpe_idx < 4 else 0.7
        with torch.no_grad():
            self.zpe_flows[zpe_idx].data = momentum * self.zpe_flows[zpe_idx].data + (1 - momentum) * (1.0 + perturbation)
            self.zpe_flows[zpe_idx].data = torch.clamp(self.zpe_flows[zpe_idx].data, 0.8, 1.2)

    def apply_zpe(self, x, zpe_idx, spatial=True):
        if x.numel() == 0:
            return x
        feature_size = x.size(1) if spatial else x.size(-1)
        if feature_size == 0 and x.numel() > 0:
            if not spatial and x.dim() == 2:
                feature_size = x.size(1)
            else:
                return x
        elif feature_size == 0 and x.numel() == 0:
            return x
        if self.sequence_length == 0:
            return x
        self.perturb_zpe_flow(x, zpe_idx, feature_size)
        flow = self.zpe_flows[zpe_idx]
        if spatial:
            if x.size(2) == 0 or x.size(3) == 0:
                return x
            num_elements_to_cover = x.size(2) * x.size(3)
            repeats = (num_elements_to_cover + self.sequence_length - 1) // self.sequence_length
            flow_expanded_flat = flow.repeat(repeats)[:num_elements_to_cover]
            flow_expanded = flow_expanded_flat.reshape(1, 1, x.size(2), x.size(3))
            if x.size(1) > 0:
                flow_expanded = flow_expanded.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        else:
            if x.size(-1) == 0:
                return x
            num_elements_to_cover = x.size(-1)
            repeats = (num_elements_to_cover + self.sequence_length - 1) // self.sequence_length
            flow_expanded_flat = flow.repeat(repeats)[:num_elements_to_cover]
            flow_expanded = flow_expanded_flat.reshape(1, -1).expand_as(x)
        return x * flow_expanded

    def forward(self, x):
        x = self.apply_zpe(x, 0)
        residual = self.shortcut1(x)
        x = self.conv1(x) + residual
        x = self.apply_zpe(x, 1)
        residual = self.shortcut2(x)
        x = self.conv2(x) + residual
        x = self.apply_zpe(x, 2)
        residual = self.shortcut3(x)
        x = self.conv3(x) + residual
        x = self.apply_zpe(x, 3)
        residual = self.shortcut4(x)
        x = self.conv4(x) + residual
        x = self.apply_zpe(x, 4)
        x = self.fc(x)
        x = self.apply_zpe(x, 5, spatial=False)
        return x

    def analyze_zpe_effect(self):
        return [torch.mean(torch.abs(flow - 1.0)).item() for flow in self.zpe_flows]

# Utility Functions
def mixup(data: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, device: str = 'cpu'):
    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data, shuffled_targets = data[indices], targets[indices]
    lam = np.random.beta(alpha, alpha)
    return lam * data + (1 - lam) * shuffled_data, targets, shuffled_targets, lam

def get_gpu_usage_info_internal() -> Dict[str, Any]:
    if not PYNVML_AVAILABLE or not pynvml:
        return {"error": "PyNVML library not available."}
    if not PYNVML_INITIALIZED_SUCCESSFULLY:
        return {"error": "PyNVML not initialized successfully."}
    try:
        num_gpus = pynvml.nvmlDeviceGetCount()
        if num_gpus == 0:
            return {"info": "No NVIDIA GPUs detected."}
        gpu_info_list = []
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power_draw_mw = None
            try:
                power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            except pynvml.NVMLError:
                pass
            fan_speed = None
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except pynvml.NVMLError:
                pass
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8') if isinstance(pynvml.nvmlDeviceGetName(handle), bytes) else pynvml.nvmlDeviceGetName(handle)
            gpu_info_list.append({
                "id": str(i),
                "name": gpu_name,
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
        return gpu_info_list[0]
    except pynvml.NVMLError as e:
        return {"error": f"NVML Error fetching GPU stats: {str(e)}"}
    except Exception as e:
        logger.error(f"Error fetching GPU stats: {e}")
        return {"error": f"Unexpected error fetching GPU stats: {str(e)}"}

def get_cpu_usage_info_detailed() -> Dict[str, Any]:
    try:
        overall_usage = psutil.cpu_percent(interval=0.1)
        per_core_usage = psutil.cpu_percent(interval=0.1, percpu=True)
        frequency = psutil.cpu_freq()
        return {
            "overall_usage_percent": float(overall_usage),
            "per_core_usage_percent": [float(c) for c in per_core_usage],
            "current_frequency_mhz": float(frequency.current) if frequency else None,
            "max_frequency_mhz": float(frequency.max) if frequency else None,
        }
    except Exception as e:
        logger.error(f"Error fetching CPU stats: {e}")
        return {"error": f"Error: {str(e)}"}

def save_job_status(job_id: str):
    if job_id not in active_jobs:
        return
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    try:
        with open(job_file, "w") as f:
            json.dump(active_jobs[job_id], f, indent=2)
    except Exception as e:
        logger.error(f"Error saving job status for {job_id}: {e}")

def load_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if os.path.exists(job_file):
        try:
            with open(job_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading job status for {job_id}: {e}")
    return None

def save_model_config(config_id: str):
    if config_id not in model_configs:
        return
    config_file = os.path.join(CONFIGS_DIR, f"{config_id}.json")
    try:
        with open(config_file, "w") as f:
            json.dump(model_configs[config_id], f, indent=2)
    except Exception as e:
        logger.error(f"Error saving model config for {config_id}: {e}")

def load_model_config(config_id: str) -> Optional[Dict[str, Any]]:
    config_file = os.path.join(CONFIGS_DIR, f"{config_id}.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model config for {config_id}: {e}")
    return None

# Dataset Preparation
def preprocess_dataset(request: DatasetPrepRequest) -> Dict[str, Any]:
    try:
        input_path = os.path.join(DATASET_DIR, request.input_file)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dataset file {input_path} not found")

        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        # Clean Text
        for col in request.feature_columns:
            if col in df.columns:
                df[col] = df[col].str.lower().str.replace(r'[^\w\s]', '', regex=True)
                df[col] = df[col].fillna("")

        # Create Target if Specified
        if request.target_column and request.target_column not in df.columns:
            df['target'] = df['repo'].apply(
                lambda x: 'quantum' if any(kw in x.lower() for kw in ['quantum', 'pyseqm']) else 'non-quantum'
            )
            target_column = 'target'
        else:
            target_column = request.target_column

        # Vectorize Features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        feature_data = vectorizer.fit_transform(df[request.feature_columns].apply(lambda x: ' '.join(x), axis=1))
        feature_df = pd.DataFrame(feature_data.toarray(), columns=vectorizer.get_feature_names_out())

        # Encode Target if Supervised
        if target_column:
            le = LabelEncoder()
            df['target_encoded'] = le.fit_transform(df[target_column])
            processed_df = pd.concat([feature_df, df['target_encoded']], axis=1)
        else:
            processed_df = feature_df

        # Save Processed Dataset
        output_path = os.path.join(DATASET_DIR, request.output_file)
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed dataset to {output_path}")

        return {
            "status": "success",
            "input_shape": df.shape,
            "output_shape": processed_df.shape,
            "features": list(feature_df.columns),
            "output_file": output_path
        }
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        return {"error": str(e)}

# Training Job
def run_training_job(job_id: str, params: TrainingParameters):
    global active_jobs
    if job_id not in active_jobs:
        loaded_status = load_job_status(job_id)
        if loaded_status:
            active_jobs[job_id] = loaded_status
        else:
            active_jobs[job_id] = {
                "job_id": job_id, "status": "failed", "log_messages": [f"Job ID {job_id} not found."],
                "parameters": params.model_dump(), "total_epochs": params.totalEpochs,
                "current_epoch": 0, "accuracy": 0.0, "loss": 0.0, "zpe_effects": [],
                "zpe_history": [], "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(), "gpu_info": get_gpu_usage_info_internal()
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
        model = ZPEDeepNet(input_channels=1, output_size=10, sequence_length=10).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=params.labelSmoothing)
        optimizer = optim.Adam(model.parameters(), lr=params.learningRate, weight_decay=params.weightDecay)
        scheduler = CosineAnnealingLR(optimizer, T_max=params.totalEpochs)
        job_status["log_messages"].append("Model & Optimizer initialized.")
        job_status["gpu_info"] = get_gpu_usage_info_internal()
        save_job_status(job_id)

        # === CONTINUE TRAINING LOGIC ===
        if hasattr(params, 'baseConfigId') and params.baseConfigId:
            prev_model_path = os.path.join(JOBS_DIR, f"{params.baseConfigId}_model.pth")
            if os.path.exists(prev_model_path):
                try:
                    model.load_state_dict(torch.load(prev_model_path, map_location=device))
                    job_status["log_messages"].append(f"Loaded previous weights from {prev_model_path} for continue training.")
                except Exception as e:
                    job_status["log_messages"].append(f"Failed to load previous weights from {prev_model_path}: {e}. Starting fresh.")
                    # List available .pth files for debugging
                    available = [f for f in os.listdir(JOBS_DIR) if f.endswith('_model.pth')]
                    job_status["log_messages"].append(f"Available model files: {available}")
            else:
                job_status["log_messages"].append(f"No previous weights found for baseConfigId {params.baseConfigId}, starting fresh.")

        for epoch in range(1, params.totalEpochs + 1):
            if job_status["status"] == "stopped":
                job_status["log_messages"].append("Training stopped.")
                break

            epoch_start_time = time.time()
            model.train()
            epoch_loss_sum, num_batches_epoch = 0.0, 0
            job_status["log_messages"].append(f"--- Epoch {epoch}/{params.totalEpochs} ---")
            job_status["gpu_info"] = get_gpu_usage_info_internal()
            save_job_status(job_id)

            for batch_idx, (data, target) in enumerate(train_loader):
                if job_status["status"] == "stopped":
                    break
                data, target = data.to(device), target.to(device)
                data_mixed, target_a, target_b, lam = mixup(data, target, alpha=0.4, device=device)
                optimizer.zero_grad()
                output = model(data_mixed)
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item()
                num_batches_epoch += 1
                if batch_idx > 0 and batch_idx % (len(train_loader) // 5 or 1) == 0:
                    job_status["log_messages"].append(f"E{epoch} B{batch_idx}/{len(train_loader)} L: {loss.item():.4f}")
                    job_status.update({
                        "current_epoch": epoch, "loss": loss.item(),
                        "zpe_effects": model.analyze_zpe_effect(),
                        "gpu_info": get_gpu_usage_info_internal(),
                    })
                    try:
                        save_job_status(job_id)
                    except Exception as e:
                        logger.error(f"Error saving job status during epoch {epoch}, batch {batch_idx}: {e}")

            scheduler.step()
            avg_epoch_train_loss = epoch_loss_sum / num_batches_epoch if num_batches_epoch > 0 else 0.0
            model.eval()
            val_correct, val_total, val_loss_sum, val_batches_epoch = 0, 0, 0.0, 0
            with torch.no_grad():
                for data_val, target_val in val_loader:
                    data_val, target_val = data_val.to(device), target_val.to(device)
                    output_val = model(data_val)
                    val_loss = criterion(output_val, target_val)
                    val_loss_sum += val_loss.item()
                    val_batches_epoch += 1
                    _, predicted_val = torch.max(output_val.data, 1)
                    val_total += target_val.size(0)
                    val_correct += (predicted_val == target_val).sum().item()

            val_accuracy = (100 * val_correct / val_total) if val_total > 0 else 0.0
            avg_epoch_val_loss = val_loss_sum / val_batches_epoch if val_batches_epoch > 0 else 0.0
            epoch_time = time.time() - epoch_start_time

            # Update job status
            job_status.update({
                "current_epoch": epoch, "accuracy": val_accuracy, "loss": avg_epoch_val_loss,
                "zpe_effects": model.analyze_zpe_effect(),
                "gpu_info": get_gpu_usage_info_internal()
            })
            job_status["zpe_history"].append({
                "epoch": epoch, "zpe_effects": job_status["zpe_effects"]
            })

            # Log epoch data to CSV
            training_logger.log_epoch(
                job_id=job_status.get("job_id", ""),
                epoch=epoch,
                total_epochs=params.totalEpochs,
                timestamp=datetime.now().isoformat(),
                model_name=params.modelName,
                base_config_id=getattr(params, "baseConfigId", ""),
                train_loss=avg_epoch_train_loss,
                val_loss=avg_epoch_val_loss,
                val_accuracy=val_accuracy,
                epoch_time=epoch_time,
                params=params.model_dump() if hasattr(params, "model_dump") else vars(params),
                zpe_effects=job_status.get("zpe_effects", [0.0]*6)
            )

            job_status["log_messages"].append(f"E{epoch} END - TrainL: {avg_epoch_train_loss:.4f}, ValAcc: {val_accuracy:.2f}%, ValL: {avg_epoch_val_loss:.4f}")
            job_status["log_messages"].append(f"ZPE: {[f'{x:.4f}' for x in job_status['zpe_effects']]}")
            save_job_status(job_id)

        if job_status["status"] != "stopped":
            job_status["status"] = "completed"
            final_msg = f"Training done! Final Val Acc: {job_status['accuracy']:.2f}%"
            job_status["log_messages"].append(final_msg)
            logger.info(f"Job {job_id}: {final_msg}")
            model_save_path = os.path.join(JOBS_DIR, f"{job_id}_model.pth")
            torch.save(model.state_dict(), model_save_path)
            job_status["log_messages"].append(f"Model saved: {model_save_path}")
            
            # Log final job completion
            training_logger.log_job_completion(job_status['job_id'], job_status['accuracy'], job_status['loss'])

    except Exception as e:
        job_status["status"] = "failed"
        error_msg = f"Train Error job {job_id}: {str(e)}"
        tb_str = traceback.format_exc()
        job_status["log_messages"].extend([error_msg, tb_str])
        logger.error(f"{error_msg}\n{tb_str}")
    finally:
        job_status["end_time"] = datetime.now().isoformat()
        job_status["gpu_info"] = get_gpu_usage_info_internal()
        save_job_status(job_id)

# Chatbot AI Flow Functions
async def call_get_quantum_explanation(job_id: str) -> str:
    job_status = active_jobs.get(job_id, load_job_status(job_id))
    if not job_status:
        return "Job not found."
    zpe_effects = job_status.get("zpe_effects", [0.0] * 6)
    return f"Quantum ZPE effects for job {job_id}: {zpe_effects}. These represent perturbations applied to model layers to simulate quantum fluctuations."

async def call_advise_hsqnn_parameters(job_id: str) -> str:
    job_status = active_jobs.get(job_id, load_job_status(job_id))
    if not job_status:
        return "Job not found."

    try:
        previous_parameters = job_status.get("parameters", {})

        # Assuming you have an AI flow function like this available
        # You might need to import it at the top of your zpe_train_api.py
        # from your_ai_flows_module import advise_hsqnn_parameters_flow

        # Call your actual AI flow implementation
        # You might need to pass the user objective here if your flow uses it
        # For simplicity, we're just passing previous_parameters
        # suggested_parameters = await advise_hsqnn_parameters_flow(previous_parameters=previous_parameters)

        # TEMPORARY PLACEHOLDER for the AI flow call
        suggested_parameters = {"learningRate": previous_parameters.get("learningRate", 0.001) * 0.9, "batchSize": min(int(previous_parameters.get("batchSize", 32) * 1.5), 128), "quantumCircuitSize": previous_parameters.get("quantumCircuitSize", 10) + 1}

        return f"Suggested HS-QNN parameters for job {job_id}:\n{json.dumps(suggested_parameters, indent=2)}"

    except Exception as e:
        logger.error(f"Error calling HS-QNN Advisor flow for job {job_id}: {e}")
        return f"An error occurred while generating parameter advice for job {job_id}. Error: {e}"

async def call_simulate_zpe(job_id: str) -> str:
    job_status = active_jobs.get(job_id, load_job_status(job_id))
    if not job_status:
        return "Job not found."
    return f"Simulated ZPE effect for job {job_id}: Perturbation strength = {np.random.normal(0.1, 0.02):.4f}"

# API Endpoints
@app.post("/api/train", summary="Start a training job")
async def start_training_endpoint(params: TrainingParameters, background_tasks: BackgroundTasks):
    job_id = f"zpe_job_{str(uuid.uuid4())[:8]}"
    active_jobs[job_id] = {
        "job_id": job_id, "status": "pending", "current_epoch": 0, "zpe_history": [],
        "total_epochs": params.totalEpochs, "accuracy": 0.0, "loss": 0.0,
        "zpe_effects": [0.0] * 6, "log_messages": [f"Init job: {params.modelName}"],
        "parameters": params.model_dump(), "start_time": None, "end_time": None,
        "gpu_info": get_gpu_usage_info_internal()
    }
    # If baseConfigId is provided, log it and ensure it is used for weight loading in run_training_job
    if hasattr(params, 'baseConfigId') and params.baseConfigId:
        active_jobs[job_id]["log_messages"].append(f"baseConfigId provided: {params.baseConfigId}. Will attempt to load weights from this job if available.")
    save_job_status(job_id)
    background_tasks.add_task(run_training_job, job_id, params)
    return {"status": "training_started", "job_id": job_id}

@app.get("/api/status/{job_id}", summary="Get job status")
async def get_job_status_endpoint(job_id: str):
    if job_id in active_jobs and active_jobs[job_id]["status"] in ["running", "pending"]:
        if active_jobs[job_id]["status"] == "running":
            active_jobs[job_id]["gpu_info"] = get_gpu_usage_info_internal()
        return active_jobs[job_id]
    job_status_from_file = load_job_status(job_id)
    if job_status_from_file:
        if job_status_from_file["status"] in ["completed", "failed", "stopped"]:
            active_jobs[job_id] = job_status_from_file
            return job_status_from_file
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

@app.post("/api/stop/{job_id}", summary="Stop a training job")
async def stop_training_endpoint(job_id: str):
    if job_id in active_jobs and active_jobs[job_id]["status"] in ["running", "pending"]:
        active_jobs[job_id]["status"] = "stopped"
        active_jobs[job_id]["log_messages"].append("Stop request received.")
        save_job_status(job_id)
        return {"status": "stop_requested", "job_id": job_id}
    job_status_from_file = load_job_status(job_id)
    if job_status_from_file:
        return {"status": job_status_from_file["status"], "message": "Job not active.", "job_id": job_id}
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

@app.get("/api/jobs", summary="List all jobs")
async def list_jobs_endpoint(limit: int = 20):
    processed_job_ids, jobs_summary_list = set(), []
    for job_id, status_dict in list(active_jobs.items()):
        model_name = status_dict.get("parameters", {}).get("modelName", "Unknown")
        jobs_summary_list.append({
            "job_id": job_id, "model_name": model_name, "status": status_dict.get("status", "unknown"),
            "accuracy": status_dict.get("accuracy", 0.0), "current_epoch": status_dict.get("current_epoch", 0),
            "total_epochs": status_dict.get("total_epochs", 0), "start_time": status_dict.get("start_time")
        })
        processed_job_ids.add(job_id)
    try:
        job_files = sorted(
            [f for f in os.listdir(JOBS_DIR) if f.endswith('.json')],
            key=lambda f: os.path.getmtime(os.path.join(JOBS_DIR, f)), reverse=True
        )
        for job_file_name in job_files:
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
    except Exception as e:
        logger.error(f"Error listing jobs from disk: {e}")

    final_jobs_map = {job['job_id']: job for job in jobs_summary_list}
    sorted_jobs = sorted(final_jobs_map.values(), key=lambda x: x.get("start_time") or "", reverse=True)
    return {"jobs": sorted_jobs[:limit]}

@app.get("/api/gpu-stats", summary="Get GPU statistics")
async def get_system_gpu_stats_endpoint():
    return get_gpu_usage_info_internal()

@app.get("/api/system-stats", summary="Get system statistics")
async def get_system_general_stats_endpoint():
    gpu_data = get_gpu_usage_info_internal()
    primary_gpu_info = gpu_data if isinstance(gpu_data, dict) else None
    return {
        "gpu_info": primary_gpu_info,
        "cpu_info": get_cpu_usage_info_detailed()
    }

@app.post("/api/configs", summary="Create a model configuration")
async def create_model_config_endpoint(config: ModelConfig):
    config_id = f"config_{str(uuid.uuid4())[:8]}"
    config.id = config_id
    model_configs[config_id] = config.model_dump()
    save_model_config(config_id)
    return {"status": "configuration_saved", "config_id": config_id}

@app.get("/api/configs/{config_id}", summary="Get a model configuration")
async def get_model_config_endpoint(config_id: str):
    if config_id in model_configs:
        return model_configs[config_id]
    config_data_from_file = load_model_config(config_id)
    if config_data_from_file:
        model_configs[config_id] = config_data_from_file
        return config_data_from_file
    raise HTTPException(status_code=404, detail=f"Model configuration {config_id} not found")

@app.get("/api/configs", summary="List all model configurations")
async def list_model_configs_endpoint(limit: int = 20):
    configs_list = list(model_configs.values())
    try:
        config_files = os.listdir(CONFIGS_DIR)
        for config_file_name in config_files:
            if config_file_name.endswith('.json'):
                config_id_from_file = config_file_name.replace(".json", "")
                if config_id_from_file not in model_configs:
                    config_data = load_model_config(config_id_from_file)
                    if config_data:
                        model_configs[config_id_from_file] = config_data
                        configs_list.append(config_data)
    except Exception as e:
        logger.error(f"Error listing model configurations from disk: {e}")
    return {"configs": configs_list[:limit]}

@app.delete("/api/configs/{config_id}", summary="Delete a model configuration")
async def delete_model_config_endpoint(config_id: str):
    if config_id in model_configs:
        del model_configs[config_id]
    config_file_path = os.path.join(CONFIGS_DIR, f"{config_id}.json")
    if os.path.exists(config_file_path):
        try:
            os.remove(config_file_path)
        except Exception as e:
            logger.error(f"Error deleting config file {config_file_path}: {e}")
            return {"status": "configuration_deleted_from_memory_only", "config_id": config_id}
        return {"status": "configuration_deleted", "config_id": config_id}
    if config_id not in model_configs and not os.path.exists(config_file_path):
        raise HTTPException(status_code=404, detail=f"Model configuration {config_id} not found to delete.")
    return {"status": "configuration_deleted_from_memory_only", "config_id": config_id}

@app.post("/api/dataset/prepare", summary="Prepare dataset for ML")
async def prepare_dataset_endpoint(request: DatasetPrepRequest):
    result = preprocess_dataset(request)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/api/chat", summary="Chat with quantum AI assistant")
async def chat_endpoint(chat_message: ChatMessage):
    message = chat_message.message.lower()
    job_id = chat_message.job_id
    response_text = "Intent not recognized. Please ask about quantum explanations, HS-QNN parameters, or ZPE simulations."

    # Basic intent detection
    if 'explain' in message or 'explanation' in message:
        if not job_id:
            response_text = "Please provide a job_id for quantum explanation."
        else:
            response_text = await call_get_quantum_explanation(job_id)
    elif 'advise parameters' in message or 'suggest parameters' in message:
        if not job_id:
            response_text = "Please provide a job_id for parameter advice."
        else:
            response_text = await call_advise_hsqnn_parameters(job_id)
    elif 'simulate zpe' in message or 'zpe effects' in message:
        if not job_id:
            response_text = "Please provide a job_id for ZPE simulation."
        else:
            response_text = await call_simulate_zpe(job_id)

    return {"response": response_text}

app.include_router(mixup_alpha_router)

# Main Entry Point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
