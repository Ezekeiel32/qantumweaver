# --- Inlined from src/lib/training_logger.py ---
import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class TrainingLogger:
    def __init__(self, csv_path: str = "training_logs.csv"):
        self.csv_path = csv_path
        self._ensure_csv_exists()
        self._processed_jobs = set()
        self._process_historical_logs()
    def _ensure_csv_exists(self):
        if not os.path.exists(self.csv_path):
            headers = [
                'Job_ID', 'Epoch', 'Total_Epochs', 'Timestamp', 'Model_Name', 'Base_Config_ID',
                'Train_Loss', 'Val_Loss', 'Val_Accuracy', 'Epoch_Time', 'Conv1_Channels',
                'Conv2_Channels', 'Conv3_Channels', 'Conv4_Channels', 'FC_Hidden', 'Dropout_FC',
                'Energy_Scale', 'Learning_Rate', 'Mixup_Alpha', 'Batch_Size', 'Momentum_Factor',
                'Label_Smoothing', 'Quantum_Circuit_Size', 'Quantum_Mode', 'Momentum_Params',
                'Strength_Params', 'Noise_Params', 'Coupling_Params', 'Cycle_Scale',
                'zpe_effects', 'Weight_Decay'
            ]
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    def _process_historical_logs(self):
        logs_dir = "logs.json"
        if not os.path.exists(logs_dir):
            return
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            self._processed_jobs = {row['Job_ID'] for row in reader}
        for filename in os.listdir(logs_dir):
            if not filename.endswith('.json'):
                continue
            job_id = filename.replace('.json', '')
            if job_id in self._processed_jobs:
                continue
            try:
                with open(os.path.join(logs_dir, filename), 'r') as f:
                    job_data = json.load(f)
                if job_data.get('status') != 'completed':
                    continue
                params = job_data.get('parameters', {})
                for log_msg in job_data.get('log_messages', []):
                    if 'E' in log_msg and 'END' in log_msg:
                        epoch_data = self._extract_epoch_data(log_msg)
                        if not epoch_data:
                            continue
                        zpe_effects = self._extract_zpe_effects(log_msg)
                        if zpe_effects is None:
                            zpe_effects = job_data.get('zpe_effects', [0.0] * 6)
                        self.log_epoch(
                            job_id=job_id,
                            epoch=epoch_data['epoch'],
                            total_epochs=params.get('totalEpochs', 0),
                            timestamp=job_data.get('start_time', datetime.now().isoformat()),
                            model_name=params.get('modelName', ''),
                            base_config_id=params.get('baseConfigId', ''),
                            train_loss=epoch_data['train_loss'],
                            val_loss=epoch_data['val_loss'],
                            val_accuracy=epoch_data['val_accuracy'],
                            epoch_time=0.0,
                            params=params,
                            zpe_effects=zpe_effects
                        )
                self._processed_jobs.add(job_id)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    def _extract_epoch_data(self, log_msg: str) -> Optional[Dict[str, Any]]:
        import re
        epoch_match = re.search(r'E(\d+)\s+END\s+-\s+TrainL:\s+([\d.]+),\s+ValAcc:\s+([\d.]+)%,\s+ValL:\s+([\d.]+)', log_msg)
        if not epoch_match:
            return None
        return {
            'epoch': int(epoch_match.group(1)),
            'train_loss': float(epoch_match.group(2)),
            'val_accuracy': float(epoch_match.group(3)),
            'val_loss': float(epoch_match.group(4))
        }
    def _extract_zpe_effects(self, log_msg: str) -> Optional[List[float]]:
        import re
        zpe_match = re.search(r'ZPE:\s+\[([\d.,\s]+)\]', log_msg)
        if not zpe_match:
            return None
        try:
            return [float(x.strip()) for x in zpe_match.group(1).split(',')]
        except:
            return None
    def log_epoch(self, job_id: str, epoch: int, total_epochs: int, timestamp: str,
                 model_name: str, base_config_id: str, train_loss: float, val_loss: float,
                 val_accuracy: float, epoch_time: float, params: Dict[str, Any],
                 zpe_effects: List[float]):
        momentum_params = params.get('momentumParams', [0.0] * 6)
        strength_params = params.get('strengthParams', [0.0] * 6)
        noise_params = params.get('noiseParams', [0.0] * 6)
        coupling_params = params.get('couplingParams', [0.0] * 6)
        for param_list in [momentum_params, strength_params, noise_params, coupling_params]:
            if len(param_list) < 6:
                param_list.extend([0.0] * (6 - len(param_list)))
        row = [
            job_id,
            epoch,
            total_epochs,
            timestamp,
            model_name,
            base_config_id,
            train_loss,
            val_loss,
            val_accuracy,
            epoch_time,
            params.get('conv1Channels', 0),
            params.get('conv2Channels', 0),
            params.get('conv3Channels', 0),
            params.get('conv4Channels', 0),
            params.get('fcHidden', 0),
            params.get('dropoutFC', 0.0),
            params.get('energyScale', 1.0),
            params.get('learningRate', 0.0),
            params.get('mixupAlpha', 0.0),
            params.get('batchSize', 0),
            params.get('momentumFactor', 0.0),
            params.get('labelSmoothing', 0.0),
            params.get('quantumCircuitSize', 0),
            params.get('quantumMode', False),
            ','.join(map(str, momentum_params)),
            ','.join(map(str, strength_params)),
            ','.join(map(str, noise_params)),
            ','.join(map(str, coupling_params)),
            params.get('cycleScale', 1.0),
            ','.join(map(str, zpe_effects)),
            params.get('weightDecay', 0.0)
        ]
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    def log_job_completion(self, job_id: str, final_accuracy: float, final_loss: float):
        pass
# --- End inlined training_logger.py ---

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import math
import subprocess
from typing import Optional
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
try:
    import git
except ImportError:
    git = None
import joblib
import tempfile
import shutil
import requests
import zipfile
import io

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
    model_name: str
    total_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    sequence_length: Optional[int] = 10
    label_smoothing: Optional[float] = 0.1
    mixup_alpha: Optional[float] = 1.0
    zpe_regularization_strength: Optional[float] = 0.001
    base_config_id: Optional[str] = None
    
    class Config:
        extra = "allow"

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

# --- NEW ZPEDeepNet Model Definition from user ---
class ZPEDeepNet(nn.Module):
    def __init__(self, output_size=10, sequence_length=32, momentum_params=None, strength_params=None, noise_params=None, coupling_params=None):
        super(ZPEDeepNet, self).__init__()
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zpe_flows = [torch.ones(sequence_length, device=self.device) for _ in range(6)]
        self.momentum_params = momentum_params or [0.9]*6
        self.strength_params = strength_params or [0.35]*6
        self.noise_params = noise_params or [0.3]*6
        self.coupling_params = coupling_params or [0.85]*6
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
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
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )

    def perturb_zpe_flow(self, data, zpe_idx, feature_size):
        with torch.no_grad():
            batch_mean = torch.mean(data.detach(), dim=0).view(-1)
            divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
            batch_mean_truncated = batch_mean[:divisible_size]
            reshaped = batch_mean_truncated.view(-1, self.sequence_length)
            perturbation = torch.mean(reshaped, dim=0)
            noise = torch.randn_like(perturbation) * self.noise_params[zpe_idx]
            perturbation = perturbation + noise
            perturbation = torch.tanh(perturbation * self.strength_params[zpe_idx])
            self.zpe_flows[zpe_idx] = (
                self.momentum_params[zpe_idx] * self.zpe_flows[zpe_idx] + 
                (1 - self.momentum_params[zpe_idx]) * (1.0 + perturbation)
            )
            self.zpe_flows[zpe_idx] = (
                self.coupling_params[zpe_idx] * self.zpe_flows[zpe_idx] + 
                (1 - self.coupling_params[zpe_idx]) * torch.ones_like(self.zpe_flows[zpe_idx])
            )
            self.zpe_flows[zpe_idx] = torch.clamp(self.zpe_flows[zpe_idx], 0.8, 1.2)

    def apply_zpe(self, x, zpe_idx, spatial=True):
        self.perturb_zpe_flow(x, zpe_idx, x.size(1) if spatial else x.size(-1))
        flow = self.zpe_flows[zpe_idx]
        if spatial:
            size = x.size(2) * x.size(3)
            flow_expanded = flow.repeat(size // self.sequence_length + 1)[:size].view(1, 1, x.size(2), x.size(3))
            flow_expanded = flow_expanded.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        else:
            flow_expanded = flow.repeat(x.size(-1) // self.sequence_length + 1)[:x.size(-1)].view(1, -1)
            flow_expanded = flow_expanded.expand(x.size(0), x.size(-1))
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

# --- END NEW ZPEDeepNet Model ---

# Utility Functions
def mixup(data: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    data = lam * data + (1 - lam) * shuffled_data
    return data, targets, shuffled_targets, lam

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

# --- NEW Training Job Logic from user ---
def safe_float(val):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return 0.0
        return float(val)
    except Exception:
        return 0.0

def run_training_job(job_id: str, params: TrainingParameters):
    global active_jobs
    if job_id not in active_jobs:
        loaded_status = load_job_status(job_id)
        if loaded_status:
            active_jobs[job_id] = loaded_status
        else:
            active_jobs[job_id] = {
                "job_id": job_id, "status": "failed", "log_messages": [f"Job ID {job_id} not found."],
                "parameters": params.model_dump(), "total_epochs": params.total_epochs,
                "current_epoch": 0, "accuracy": 0.0, "loss": 0.0, "zpe_effects": [],
                "zpe_history": [], "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(), "gpu_info": get_gpu_usage_info_internal()
            }
            save_job_status(job_id)
            return

    job_status = active_jobs[job_id]
    job_status["status"] = "running"
    job_status["log_messages"].append(f"Starting PyTorch training: {params.model_name}")
    job_status["start_time"] = datetime.now().isoformat()
    job_status["gpu_info"] = get_gpu_usage_info_internal()
    save_job_status(job_id)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        job_status["log_messages"].append(f"Device: {device}")
        job_status["log_messages"].append("Setting up data transforms...")
        train_transform = transforms.Compose([
            transforms.RandomRotation(15), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomCrop(28, padding=2), transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        ])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset_full = datasets.MNIST(root=DATASET_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=DATASET_DIR, train=False, download=True, transform=test_transform)
        train_loader = DataLoader(train_dataset_full, batch_size=params.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))
        job_status["log_messages"].append(f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)} batches.")
        model = ZPEDeepNet(output_size=10, sequence_length=params.sequence_length).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=params.total_epochs)
        job_status["log_messages"].append("Model & Optimizer initialized.")
        job_status["gpu_info"] = get_gpu_usage_info_internal()
        save_job_status(job_id)

        if hasattr(params, 'base_config_id') and params.base_config_id:
            prev_model_path = os.path.join(JOBS_DIR, f"{params.base_config_id}_model.pth")
            if os.path.exists(prev_model_path):
                try:
                    model.load_state_dict(torch.load(prev_model_path, map_location=device))
                    job_status["log_messages"].append(f"Loaded previous weights from {prev_model_path} for continue training.")
                except Exception as e:
                    job_status["log_messages"].append(f"Failed to load previous weights from {prev_model_path}: {e}. Starting fresh.")
                    available = [f for f in os.listdir(JOBS_DIR) if f.endswith('_model.pth')]
                    job_status["log_messages"].append(f"Available model files: {available}")
            else:
                job_status["log_messages"].append(f"No previous weights found for base_config_id {params.base_config_id}, starting fresh.")

        for epoch in range(1, params.total_epochs + 1):
            if job_status["status"] == "stopped":
                job_status["log_messages"].append("Training stopped.")
                break
            model.train()
            epoch_loss_sum, num_batches_epoch = 0.0, 0
            job_status["log_messages"].append(f"--- Epoch {epoch}/{params.total_epochs} ---")
            job_status["gpu_info"] = get_gpu_usage_info_internal()
            save_job_status(job_id)

            for batch_idx, (data, target) in enumerate(train_loader):
                if job_status["status"] == "stopped":
                    break
                data, target = data.to(device), target.to(device)
                data_mixed, target_a, target_b, lam = mixup(data, target, alpha=params.mixup_alpha)
                optimizer.zero_grad()
                output = model(data_mixed)
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                # Skip NaN/Inf losses
                if torch.isnan(loss) or torch.isinf(loss):
                    job_status["log_messages"].append(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Skipping batch.")
                    continue
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item()
                num_batches_epoch += 1
                if batch_idx > 0 and batch_idx % (len(train_loader) // 5 or 1) == 0:
                    job_status["log_messages"].append(f"E{epoch} B{batch_idx}/{len(train_loader)} L: {safe_float(loss.item()):.4f}")
                    job_status.update({
                        "current_epoch": epoch, "loss": safe_float(loss.item()),
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
                    # Skip NaN/Inf val losses
                    if torch.isnan(val_loss) or torch.isinf(val_loss):
                        job_status["log_messages"].append(f"NaN/Inf val_loss at epoch {epoch}. Skipping batch.")
                        continue
                    val_loss_sum += val_loss.item()
                    val_batches_epoch += 1
                    _, predicted_val = torch.max(output_val.data, 1)
                    val_total += target_val.size(0)
                    val_correct += (predicted_val == target_val).sum().item()
            val_accuracy = (100 * val_correct / val_total) if val_total > 0 else 0.0
            avg_epoch_val_loss = val_loss_sum / val_batches_epoch if val_batches_epoch > 0 else 0.0
            train_accuracy = (100 * val_correct / val_total) if val_total > 0 else 0.0
            job_status.update({
                "current_epoch": epoch, "accuracy": safe_float(val_accuracy), "loss": safe_float(avg_epoch_val_loss),
                "zpe_effects": model.analyze_zpe_effect(),
                "gpu_info": get_gpu_usage_info_internal()
            })
            job_status["zpe_history"].append({
                "epoch": epoch, "zpe_effects": job_status["zpe_effects"]
            })
            metrics = {
                "epoch": epoch,
                "train_loss": safe_float(avg_epoch_train_loss),
                "train_accuracy": safe_float(train_accuracy),
                "val_loss": safe_float(avg_epoch_val_loss),
                "val_accuracy": safe_float(val_accuracy)
            }
            if "metrics" not in job_status:
                job_status["metrics"] = []
            job_status["metrics"].append(metrics)
            job_status["log_messages"].append(f"E{epoch} END - TrainL: {safe_float(avg_epoch_train_loss):.4f}, ValAcc: {safe_float(val_accuracy):.2f}%, ValL: {safe_float(avg_epoch_val_loss):.4f}")
            job_status["log_messages"].append(f"ZPE: {[f'{x:.4f}' for x in job_status['zpe_effects']]}")
            save_job_status(job_id)

        if job_status["status"] != "stopped":
            job_status["status"] = "completed"
            final_msg = f"Training done! Final Val Acc: {safe_float(job_status['accuracy']):.2f}%"
            job_status["log_messages"].append(final_msg)
            logger.info(f"Job {job_id}: {final_msg}")
            model_save_path = os.path.join(JOBS_DIR, f"{job_id}_model.pth")
            torch.save(model.state_dict(), model_save_path)
            job_status["log_messages"].append(f"Model saved: {model_save_path}")
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

# --- END NEW Training Job Logic ---

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
        suggested_parameters = {"learning_rate": previous_parameters.get("learning_rate", 0.001) * 0.9, "batch_size": min(int(previous_parameters.get("batch_size", 32) * 1.5), 128), "quantum_circuit_size": previous_parameters.get("quantum_circuit_size", 10) + 1}

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
    logger.info(f"/api/train called with parameters: {params.model_dump_json()}")
    try:
        job_id = f"zpe_job_{str(uuid.uuid4())[:8]}"
        active_jobs[job_id] = {
            "job_id": job_id, "status": "pending", "current_epoch": 0,
            "total_epochs": params.total_epochs, "accuracy": 0.0, "loss": 0.0,
            "zpe_effects": [0.0] * 6, "log_messages": [f"Init job: {params.model_name}"],
            "parameters": params.model_dump(), "start_time": None, "end_time": None,
            "gpu_info": get_gpu_usage_info_internal(), "metrics_history": []
        }
        
        if params.base_config_id:
            active_jobs[job_id]["log_messages"].append(f"Attempting to use base_config_id: {params.base_config_id}.")
        
        save_job_status(job_id)
        background_tasks.add_task(run_training_job, job_id, params)
        return {"status": "training_started", "job_id": job_id}
    except Exception as e:
        logger.error(f"Error in /api/train: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

@app.get("/api/status/{job_id}", summary="Get job status")
async def get_job_status_endpoint(job_id: str):
    try:
        if job_id in active_jobs and active_jobs[job_id]["status"] in ["running", "pending"]:
            if active_jobs[job_id]["status"] == "running":
                active_jobs[job_id]["gpu_info"] = get_gpu_usage_info_internal()
                job_status = active_jobs[job_id]
                # Always provide a 'metrics' field for frontend
                if "metrics" in job_status:
                    job_status["metrics"] = job_status["metrics"]
                elif "metrics_history" in job_status:
                    job_status["metrics"] = job_status["metrics_history"]
                return job_status
        job_status_from_file = load_job_status(job_id)
        if job_status_from_file:
            if job_status_from_file["status"] in ["completed", "failed", "stopped"]:
                active_jobs[job_id] = job_status_from_file
                if "metrics" in job_status_from_file:
                    job_status_from_file["metrics"] = job_status_from_file["metrics"]
                elif "metrics_history" in job_status_from_file:
                    job_status_from_file["metrics"] = job_status_from_file["metrics_history"]
                return job_status_from_file
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Error in /api/status/{job_id}: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

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
        model_name = status_dict.get("parameters", {}).get("model_name", "Unknown")
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
                    model_name_file = job_data.get("parameters", {}).get("model_name", "Unknown")
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

@app.post("/api/recommend_mixup_alpha")
async def recommend_mixup_alpha(request: Request):
    config = await request.json()
    features = np.array([config[feat] for feat in mixup_alpha_model.feature_names_in_]).reshape(1, -1)
    pred = mixup_alpha_model.predict(features)[0]
    return {"predicted_accuracy": float(pred)}

@app.post("/api/sweep_mixup_alpha")
async def sweep_mixup_alpha(request: Request):
    config = await request.json()
    candidate_alphas = np.arange(0.0, 2.1, 0.1)
    best_alpha = None
    best_pred = -np.inf
    for alpha in candidate_alphas:
        config['Mixup_Alpha'] = alpha
        features = np.array([config[feat] for feat in mixup_alpha_model.feature_names_in_]).reshape(1, -1)
        pred = mixup_alpha_model.predict(features)[0]
        if pred > best_pred:
            best_pred = pred
            best_alpha = alpha
    return {"recommended_mixup_alpha": best_alpha, "predicted_accuracy": float(best_pred)}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": exc.errors()},
    )

@app.get("/api/active-job", summary="Get the most recent active job ID")
async def get_active_job_endpoint():
    # Find the most recent job with status running or pending
    active = [job for job in active_jobs.values() if job.get("status") in ["running", "pending"]]
    if not active:
        return {"job_id": None}
    # Sort by start_time (descending)
    active_sorted = sorted(active, key=lambda x: x.get("start_time") or "", reverse=True)
    return {"job_id": active_sorted[0]["job_id"]}

@app.post("/api/datasets/download")
async def download_dataset(request: Request):
    data = await request.json()
    source = data.get("source")
    identifier = data.get("identifier")
    dest_dir = data.get("dest_dir")
    if not (source and identifier and dest_dir):
        return {"status": "error", "message": "Missing required fields."}
    if source == "kaggle":
        result = download_kaggle_dataset(identifier, dest_dir, data.get("kaggle_username"), data.get("kaggle_key"))
    elif source == "huggingface":
        result = download_hf_dataset(identifier, dest_dir, data.get("split", "train"))
    elif source == "github":
        result = download_github_repo(identifier, dest_dir)
    else:
        result = {"status": "error", "message": "Unknown source."}
    return result

def download_kaggle_dataset(identifier: str, dest_dir: str, username: str = None, key: str = None):
    try:
        # Set up Kaggle credentials if provided
        if username and key:
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
        # Install Kaggle API if not available
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            subprocess.run(["pip", "install", "kaggle"], check=True)
            from kaggle.api.kaggle_api_extended import KaggleApi
        # Authenticate and download dataset
        api = KaggleApi()
        api.authenticate()
        # Create destination directory
        full_dest = os.path.join(DATASET_DIR, dest_dir, identifier.replace("/", "_"))
        os.makedirs(full_dest, exist_ok=True)
        # Download dataset
        api.dataset_download_files(identifier, path=full_dest, unzip=True)
        return {"status": "success", "message": f"Dataset downloaded to {full_dest}"}
    except Exception as e:
        return {"status": "error", "message": f"Kaggle download failed: {str(e)}"}

def download_hf_dataset(identifier: str, dest_dir: str, split: str = "train"):
    try:
        # Install datasets library if not available
        try:
            from datasets import load_dataset
        except ImportError:
            subprocess.run(["pip", "install", "datasets"], check=True)
            from datasets import load_dataset
        # Create destination directory
        full_dest = os.path.join(DATASET_DIR, dest_dir, identifier)
        os.makedirs(full_dest, exist_ok=True)
        # Download dataset
        dataset = load_dataset(identifier, split=split)
        # Save dataset
        dataset.save_to_disk(full_dest)
        return {"status": "success", "message": f"Dataset downloaded to {full_dest}"}
    except Exception as e:
        return {"status": "error", "message": f"Hugging Face download failed: {str(e)}"}

def download_github_repo(url: str, dest_dir: str):
    try:
        # Extract repo name from URL
        repo_name = url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        # Create destination directory
        full_dest = os.path.join(DATASET_DIR, dest_dir, repo_name)
        os.makedirs(full_dest, exist_ok=True)
        # Download as zip archive
        if not url.startswith("https://github.com/"):
            return {"status": "error", "message": "Invalid GitHub URL"}
        zip_url = url.replace("github.com", "codeload.github.com") + "/zip/main"
        response = requests.get(zip_url)
        if response.status_code != 200:
            # Try master branch if main doesn't work
            zip_url = zip_url.replace("/main", "/master")
            response = requests.get(zip_url)
            if response.status_code != 200:
                return {"status": "error", "message": f"Failed to download repo (status {response.status_code})"}
        # Extract zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Extract to temp directory first
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_ref.extractall(tmpdir)
                # Find the extracted folder (should be the only one)
                extracted_folder = os.path.join(tmpdir, os.listdir(tmpdir)[0])
                # Move to final destination
                shutil.move(extracted_folder, full_dest)
        return {"status": "success", "message": f"Repository cloned to {full_dest}"}
    except Exception as e:
        return {"status": "error", "message": f"GitHub download failed: {str(e)}"}

@app.post("/api/projects")
async def create_project(request: Request):
    data = await request.json()
    # Add a fake id for now
    data["id"] = "project_" + str(abs(hash(data.get("name", ""))))
    return JSONResponse(content=data)

# Main Entry Point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
