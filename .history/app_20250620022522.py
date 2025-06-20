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
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

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
    total_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    momentum_params: List[float]
    strength_params: List[float]
    noise_params: List[float]
    coupling_params: Optional[List[float]] = None
    quantum_circuit_size: int
    label_smoothing: float
    quantum_mode: bool
    model_name: str
    base_config_id: Optional[str] = None
    mixup_alpha: Optional[float] = 0.2
    channel_sizes: Optional[List[int]] = None
    class Config:
        extra = "allow"
        allow_population_by_field_name = True
        alias_generator = lambda s: s.lower() if s.islower() else ''.join(['_' + c.lower() if c.isupper() else c for c in s]).lstrip('_')

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
    def __init__(self, output_size=10, sequence_length=10):
        super(ZPEDeepNet, self).__init__()
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zpe_flows = [torch.ones(sequence_length, device=self.device) for _ in range(6)]

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
        batch_mean = torch.mean(data.detach(), dim=0).view(-1)
        if self.sequence_length == 0: return # Avoid division by zero
        divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
        if divisible_size == 0: return # Not enough data
        batch_mean_truncated = batch_mean[:divisible_size]
        reshaped = batch_mean_truncated.view(-1, self.sequence_length)
        perturbation = torch.mean(reshaped, dim=0)
        perturbation = torch.tanh(perturbation * 0.3)
        momentum = 0.9 if zpe_idx < 4 else 0.7
        with torch.no_grad():
            self.zpe_flows[zpe_idx] = momentum * self.zpe_flows[zpe_idx] + (1 - momentum) * (1.0 + perturbation)
            self.zpe_flows[zpe_idx] = torch.clamp(self.zpe_flows[zpe_idx], 0.8, 1.2)

    def apply_zpe(self, x, zpe_idx, spatial=True):
        if self.sequence_length == 0: return x # Avoid division by zero
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
def run_training_job(job_id: str, params: TrainingParameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_jobs[job_id]["status"] = "running"
    active_jobs[job_id]["start_time"] = time.time()
    
    try:
        # Data Setup
        active_jobs[job_id]["log_messages"].append("Setting up data transforms...")
        save_job_status(job_id)
        train_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        active_jobs[job_id]["log_messages"].append("Loading MNIST dataset...")
        save_job_status(job_id)
        # Use DATASET_DIR constant
        train_dataset = datasets.MNIST(root=DATASET_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=DATASET_DIR, train=False, download=True, transform=test_transform)

        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=params.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=params.batch_size, shuffle=False, num_workers=2)
        
        # Model Setup
        active_jobs[job_id]["log_messages"].append("Initializing ZPEDeepNet model...")
        save_job_status(job_id)
        model = ZPEDeepNet(output_size=10, sequence_length=getattr(params, 'sequence_length', 10)).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=getattr(params, 'label_smoothing', 0.1))
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=params.total_epochs)

        # Training Loop
        for epoch in range(params.total_epochs):
            active_jobs[job_id]["current_epoch"] = epoch + 1
            model.train()
            epoch_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                mixup_alpha = getattr(params, 'mixup_alpha', 1.0)
                data, target_a, target_b, lam = mixup(data, target, alpha=mixup_alpha)
                
                optimizer.zero_grad()
                output = model(data)
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                
                zpe_effects = model.analyze_zpe_effect()
                # Assuming zpe_regularization_strength is a parameter you might want to add
                zpe_reg_strength = getattr(params, 'zpe_regularization_strength', 0.001)
                total_loss = loss + zpe_reg_strength * sum(zpe_effects)
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            scheduler.step()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Update job status for frontend
            log_msg = f"Epoch {epoch+1}/{params.total_epochs}, Loss: {avg_epoch_loss:.4f}, Val Acc: {val_acc:.2f}%"
            active_jobs[job_id]["log_messages"].append(log_msg)
            active_jobs[job_id]["loss"] = avg_epoch_loss
            active_jobs[job_id]["accuracy"] = val_acc
            active_jobs[job_id]["zpe_effects"] = zpe_effects
            
            # For chart data
            epoch_metrics = {
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "val_loss": avg_val_loss,
                "accuracy": val_acc,
                "val_acc": val_acc # Use val_acc for both for now
            }
            if "metrics_history" not in active_jobs[job_id]:
                active_jobs[job_id]["metrics_history"] = []
            active_jobs[job_id]["metrics_history"].append(epoch_metrics)
            
            save_job_status(job_id)

        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["log_messages"].append("Training completed successfully.")
        
        # Save final model
        model_path = os.path.join(JOBS_DIR, f"{job_id}_model.pth")
        torch.save(model.state_dict(), model_path)
        active_jobs[job_id]["log_messages"].append(f"Model saved to {model_path}")

    except Exception as e:
        tb_str = traceback.format_exc()
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["log_messages"].append(f"Error during training: {e}")
        active_jobs[job_id]["log_messages"].append(tb_str)
        logger.error(f"Job {job_id} failed: {e}\n{tb_str}")
    finally:
        active_jobs[job_id]["end_time"] = time.time()
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
    print("[DEBUG] /api/train called. Params received:", params)
    try:
    job_id = f"zpe_job_{str(uuid.uuid4())[:8]}"
    active_jobs[job_id] = {
        "job_id": job_id, "status": "pending", "current_epoch": 0, "zpe_history": [],
            "total_epochs": params.total_epochs, "accuracy": 0.0, "loss": 0.0,
            "zpe_effects": [0.0] * 6, "log_messages": [f"Init job: {params.model_name}"],
        "parameters": params.model_dump(), "start_time": None, "end_time": None,
        "gpu_info": get_gpu_usage_info_internal()
    }
        # If base_config_id is provided, log it and ensure it is used for weight loading in run_training_job
        if hasattr(params, 'base_config_id') and params.base_config_id:
            active_jobs[job_id]["log_messages"].append(f"base_config_id provided: {params.base_config_id}. Will attempt to load weights from this job if available.")
    save_job_status(job_id)
    background_tasks.add_task(run_training_job, job_id, params)
    return {"status": "training_started", "job_id": job_id}
    except Exception as e:
        logger.error(f"Error in /api/train: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

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

app.include_router(mixup_alpha_router)

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

# Main Entry Point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
