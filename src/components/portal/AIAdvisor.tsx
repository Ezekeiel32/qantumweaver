import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Play, Settings, Zap, Brain, Download, Upload, Rocket, Target, GitMerge, Cpu, Code, Eye, Lightbulb, X, CheckCircle, ArrowRight, Database, ExternalLink } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import { InvokeLLM, UploadFile } from "@/integrations/Core";
import CodeModal from "./CodeModal";
import DatasetCard from "./DatasetCard";

const taskCategories = [
  {
    group: "Computer Vision",
    tasks: [
      { label: "Image Classification", value: "image_classification" },
      { label: "Image Generation", value: "image_generation" },
      { label: "Image Segmentation", value: "image_segmentation" },
      { label: "Object Detection", value: "object_detection" },
      { label: "Face Recognition", value: "face_recognition" },
      { label: "Style Transfer", value: "style_transfer" },
      { label: "Super Resolution", value: "super_resolution" },
      { label: "Pose Estimation", value: "pose_estimation" },
      { label: "Gesture Recognition", value: "gesture_recognition" },
      { label: "OCR (Text Recognition)", value: "ocr" },
      { label: "Document Analysis", value: "document_analysis" },
    ],
  },
  {
    group: "Natural Language Processing",
    tasks: [
      { label: "Text Classification", value: "text_classification" },
      { label: "Text Generation", value: "text_generation" },
      { label: "Language Translation", value: "language_translation" },
      { label: "Sentiment Analysis", value: "sentiment_analysis" },
      { label: "Named Entity Recognition", value: "ner" },
      { label: "Question Answering", value: "question_answering" },
      { label: "Text Summarization", value: "text_summarization" },
      { label: "Chatbot/Conversational AI", value: "chatbot" },
    ],
  },
  {
    group: "Audio & Speech",
    tasks: [
      { label: "Speech Recognition", value: "speech_recognition" },
      { label: "Speech Synthesis (TTS)", value: "speech_synthesis" },
      { label: "Music Generation", value: "music_generation" },
      { label: "Audio Classification", value: "audio_classification" },
      { label: "Sound Enhancement", value: "sound_enhancement" },
    ],
  },
  {
    group: "Video & Motion",
    tasks: [
      { label: "Video Classification", value: "video_classification" },
      { label: "Video Generation", value: "video_generation" },
      { label: "Deepfake Detection", value: "deepfake_detection" },
    ],
  },
  {
    group: "Data Science & Analytics",
    tasks: [
      { label: "Time Series Forecasting", value: "time_series_forecasting" },
      { label: "Anomaly Detection", value: "anomaly_detection" },
      { label: "Recommendation System", value: "recommendation_system" },
      { label: "Tabular Classification", value: "tabular_classification" },
      { label: "Tabular Regression", value: "tabular_regression" },
      { label: "Clustering", value: "clustering" },
      { label: "Financial Prediction", value: "financial_prediction" },
      { label: "Fraud Detection", value: "fraud_detection" },
    ],
  },
  {
    group: "Advanced AI",
    tasks: [
      { label: "Reinforcement Learning", value: "reinforcement_learning" },
      { label: "Game AI", value: "game_ai" },
      { label: "Autonomous Driving", value: "autonomous_driving" },
      { label: "Medical Diagnosis", value: "medical_diagnosis" },
      { label: "Drug Discovery", value: "drug_discovery" },
    ],
  },
];

interface AIAdvisorProps {
  onClose: () => void;
}

interface ProjectData {
  goal: string;
  taskTypes: string[];
  selectedDatasets: any[];
  dataStrategies: string[];
  customDataFile: any;
  dataSynthesisPrompt: string;
  numClasses: string;
  accuracyTarget: string;
  textStyle: string;
  imageResolution: string;
  objectSize: string;
  realTimeRequirement: string;
  audioLength: string;
  dataFormat: string;
  modelStrategy: string;
  mergeModelSuggestion: string;
  computeChoice: string;
  productionScale: string;
}

async function createProject(project: any) {
  const res = await fetch('/api/projects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(project),
  });
  if (!res.ok) throw new Error('Failed to create project');
  return await res.json();
}

export default function AIAdvisor({ onClose }: AIAdvisorProps) {
  const [step, setStep] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [projectData, setProjectData] = useState<ProjectData>({
    goal: "",
    taskTypes: [],
    selectedDatasets: [],
    dataStrategies: [],
    customDataFile: null,
    dataSynthesisPrompt: "",
    numClasses: "",
    accuracyTarget: "",
    textStyle: "",
    imageResolution: "",
    objectSize: "",
    realTimeRequirement: "",
    audioLength: "",
    dataFormat: "",
    modelStrategy: "scratch",
    mergeModelSuggestion: "",
    computeChoice: "cloud",
    productionScale: "1"
  });
  const [aiSuggestedDatasets, setAiSuggestedDatasets] = useState<any[]>([]);
  const [recommendations, setRecommendations] = useState<any>(null);
  const [finalTrainingParams, setFinalTrainingParams] = useState({});
  const [zpeTrainingCode, setZpeTrainingCode] = useState("");
  const [isCodeModalOpen, setIsCodeModalOpen] = useState(false);
  const [datasetSearchLoading, setDatasetSearchLoading] = useState(false);
  const [datasetSearchError, setDatasetSearchError] = useState("");

  const steps = [
    { title: "Goal", icon: Target },
    { title: "Task", icon: Zap },
    { title: "Datasets", icon: Download },
    { title: "Custom Data", icon: Upload },
    { title: "Task Specifics", icon: Settings },
    { title: "Model Strategy", icon: GitMerge },
    { title: "Compute & Scale", icon: Cpu },
    { title: "AI Review", icon: Brain },
    { title: "AI Blueprint", icon: Code }
  ];

  const handleTaskToggle = (taskValue: string) => {
    setProjectData(pd => {
      const newTasks = pd.taskTypes.includes(taskValue)
        ? pd.taskTypes.filter(t => t !== taskValue)
        : [...pd.taskTypes, taskValue];
      return { ...pd, taskTypes: newTasks };
    });
  };

  const handleDataStrategyToggle = (strategy: string) => {
    setProjectData(pd => {
      const newStrategies = pd.dataStrategies.includes(strategy)
        ? pd.dataStrategies.filter(s => s !== strategy)
        : [...pd.dataStrategies, strategy];
      return { ...pd, dataStrategies: newStrategies };
    });
  };

  const handleDatasetSelect = (datasetToToggle: any) => {
    setProjectData(currentData => {
      const isAlreadySelected = currentData.selectedDatasets.some(
        d => d.identifier === datasetToToggle.identifier
      );

      const updatedSelectedDatasets = isAlreadySelected
        ? currentData.selectedDatasets.filter(d => d.identifier !== datasetToToggle.identifier)
        : [...currentData.selectedDatasets, datasetToToggle];

      return {
        ...currentData,
        selectedDatasets: updatedSelectedDatasets,
      };
    });
  };

  const fetchDatasetsFromAPI = async () => {
    setDatasetSearchLoading(true);
    setDatasetSearchError("");
    try {
      const res = await fetch("/api/datasets/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: projectData.goal, tasks: projectData.taskTypes })
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setAiSuggestedDatasets(data.datasets || []);
    } catch (err) {
      setDatasetSearchError("Failed to fetch datasets. Try again later.");
      setAiSuggestedDatasets([]);
    }
    setDatasetSearchLoading(false);
  };

  const handleGetFinalRecommendations = async () => {
    setIsLoading(true);
    let modelMergePrompt = "";
    if (projectData.modelStrategy === 'merge') {
        const mergeSuggestions = await InvokeLLM({
            prompt: `Based on the project goal "${projectData.goal}" and task types "${projectData.taskTypes.join(', ')}", suggest the best single open-source model from HuggingFace to use as a base for fine-tuning. Provide only the model identifier (e.g., 'bert-base-uncased').`
        });
        modelMergePrompt = `The user wants to merge with a base model. The suggested model is: ${mergeSuggestions}. Consider this in your architecture recommendation.`;
        setProjectData(pd => ({...pd, mergeModelSuggestion: mergeSuggestions}));
    }

    const response = await InvokeLLM({
      prompt: `You are THE worlds' absolute expert at ML/AI Engineering. You are currently working for me and you are in my no-code,
       ML/AI building platform called ZPE.
       You are a part of a series of advisors that guide the user through the process of building their ML/AI project, 
       that form a sequence called HS-QNN (Hilbert Space Quantum Neural Network). The user has provided the following project requirements. 
       Your task is to generate the FULL BLUEPRINT of every detail of the AI building process and load the suggested configurations, datasets,
       zpe parameters, and other parameters, and all other necessary variables including the model architecture, training parameters, and zpe pytorch code.
       You are the wolds best pytorch engineer and you are also an expert in the field of pytorch.
       You are also an expert in the field of pytorch lightning and you are also an expert in the field of pytorch geometric.
       You are also an expert in the field of pytorch vision and you are also an expert in the field of pytorch audio.
       You are also an expert in the field of pytorch text and you are also an expert in the field of pytorch video.
       You are also an expert in the field of pytorch data and you are also an expert in the field of pytorch metrics.
       You are also an expert in the field of pytorch optimizers and you are also an expert in the field of pytorch loss functions.
       You are also an expert in the field of pytorch metrics and you are also an expert in the field of pytorch data.
       You are also an expert in the field of Quantum Computing and Quantum Machine Learning. 
       You are also an expert in the field of Neural Networks and Deep Learning.
       You are also an expert in the field of Computer Vision and Natural Language Processing.
       You are also an expert in the field of Audio and Speech Processing.
       You are also an expert in the field of Video and Motion Processing.
       You are also an expert in the field of Data Science and Analytics.
       You are also an expert in the field of Advanced AI.
       You are also an expert in the field of Reinforcement Learning and Game AI.
       You are also an expert in the field of Autonomous Driving and Medical Diagnosis.
       You are also an expert in the field of Drug Discovery and Financial Prediction.
       You are also an expert in the field of Fraud Detection and Anomaly Detection.
       You are also an expert in the field of Recommendation System and Tabular Classification.
       You are also an expert in the field of Tabular Regression and Clustering.
       You are also an expert in the field of Financial Prediction and Fraud Detection.
       You are also an expert in the field of Anomaly Detection and Recommendation System.
       You are also an expert in the field of Tabular Classification and Tabular Regression.
       You are also an expert in the field of Clustering and Financial Prediction.
       You are also an expert in the field of Fraud Detection and Anomaly Detection.
       You are also an expert in the field of Recommendation System and Tabular Classification.
       You are also an expert in the field of Tabular Regression and Clustering.
       You are also an expert in the field of Financial Prediction and Fraud Detection.
       You are also an expert in the field of Anomaly Detection and Recommendation System.
       You are also an expert in the field of Tabular Classification and Tabular Regression.
       You are also an expert in the field of Clustering and Financial Prediction.
       You are also an expert in the field of Fraud Detection and Anomaly Detection.
       You are also an expert in the field of batch size detection and batch size optimization.
       You are also an expert in the field of learning rate detection and learning rate optimization.
       You are also an expert in the field of optimizer detection and optimizer optimization.
       You are also an expert in the field of loss function detection and loss function optimization.
       You are also an expert in the field of metric detection and metric optimization.
       You are also an expert in the field of data augmentation detection and data augmentation optimization.
       You are also an expert in the field of data preprocessing detection and data preprocessing optimization.
       You are also an expert Quantum Physicist.
       You are also an expert in Sacred Geometry.
       You are also an expert in the field of Quantum Entanglement.
       You are also an expert in the field of Quantum Computing.
       You are also an expert in the field of Kabbalah.
       You are also an expert in the field of the Tree of Life.
       You are also an expert in the field of the Sefer Yetzirah.
       You are also an expert in the field of Quantum Physics.
       You are also an expert in the field of Quantum Mechanics.
       You are also an expert in the field of Quantum Field Theory.
       You are also an expert in the field of the ZPE platform.
       You are also an expert in the field of Task Recognition from the user's project goal and task types and data.
       You are also an expert in the field of Data Synthesis from the user's project goal and task types and data.
       You are also an expert in the field of Data Preprocessing from the user's project goal and task types and data.
       You are also an expert in the field of Data Augmentation from the user's project goal and task types and data.
       You are also an expert in the field of Data Loading from the user's project goal and task types and data.
       You are also an expert in the field of Data Validation from the user's project goal and task types and data.
       You are also an expert in the field of Data Metrics from the user's project goal and task types and data.
       You are also an expert in the field of Data Visualization from the user's project goal and task types and data.
       You are also an expert in the field of Data Analysis from the user's project goal and task types and data.
       You are also an expert in the field of Data Mining from the user's project goal and task types and data.
       You will make sure that the blueprint is provided in a JSON that fits with the training monitor.
       - Project Goal: ${projectData.goal}
      - Task Types: ${projectData.taskTypes.join(', ')}
      - Selected Datasets: ${projectData.selectedDatasets.map(d => d.name).join(', ')}
      - Task Specifics: Number of classes: ${projectData.numClasses || 'N/A'}, Text Style: ${projectData.textStyle || 'N/A'}, Object Size: ${projectData.objectSize || 'N/A'}, Image Resolution: ${projectData.imageResolution || 'N/A'}, Audio Length: ${projectData.audioLength || 'N/A'}, Data Format: ${projectData.dataFormat || 'N/A'}, Real-time: ${projectData.realTimeRequirement || 'N/A'}, Accuracy Target: ${projectData.accuracyTarget || 'N/A'}
      - Production Scale: ${projectData.productionScale} users.
      - Custom Data Strategies: ${projectData.dataStrategies.join(', ') || 'None'}
      - Custom Data File: ${projectData.customDataFile ? 'Provided' : 'Not provided'}
      - Data Synthesis Prompt: ${projectData.dataSynthesisPrompt || 'None'}
      - Model Strategy: ${projectData.modelStrategy}. ${modelMergePrompt}
      - Compute: ${projectData.computeChoice}

      Based on this, provide:
      1. A final recommended model architecture (e.g., "ZPE-Enhanced ResNet-18").
      2. A complete set of ZPE-specific training parameters.
      3. A recommended GPU for training (e.g., 'NVIDIA T4', 'NVIDIA A100').
      `,
      response_json_schema: {
        type: "object",
        properties: {
          architecture: {
            type: "object",
            properties: {
              recommended: { type: "string" },
              reasoning: { type: "string" }
            }
          },
          training_parameters: {
            type: "object",
            properties: {
              total_epochs: { type: "number" },
              batch_size: { type: "number" },
              learning_rate: { type: "number" },
              dropout_fc: { type: "number" },
              zpe_regularization_strength: { type: "number" },
              quantum_circuit_size: { type: "number" },
              quantum_mode: { type: "boolean" }
            }
          },
          gpu_recommendation: { type: "string" }
        }
      }
    });

    setRecommendations(response);
    setFinalTrainingParams(response.training_parameters);
    setIsLoading(false);
    setStep(s => s + 1);
  };

  const handleGenerateCodeAndBlueprint = async () => {
    setIsLoading(true);
    
    const zpeTemplate = `# ZPE-Enhanced PyTorch Training Script                                                                                                                        import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# ZPEDeepNet Definition
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
        divisible_size = (batch_mean.size(0) // self.sequence_length) * self.sequence_length
        batch_mean_truncated = batch_mean[:divisible_size]
        reshaped = batch_mean_truncated.view(-1, self.sequence_length)
        perturbation = torch.mean(reshaped, dim=0)
        perturbation = torch.tanh(perturbation * 0.3)
        momentum = 0.9 if zpe_idx < 4 else 0.7
        with torch.no_grad():
            self.zpe_flows[zpe_idx] = momentum * self.zpe_flows[zpe_idx] + (1 - momentum) * (1.0 + perturbation)
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

# Data Setup
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

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# MixUp Function
def mixup(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    data = lam * data + (1 - lam) * shuffled_data
    return data, targets, shuffled_targets, lam

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ZPEDeepNet(output_size=10).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=30)

# Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target_a, target_b, lam = mixup(data, target)
        optimizer.zero_grad()
        output = model(data)
        loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
        zpe_effects = model.analyze_zpe_effect()
        total_loss = loss + 0.001 * sum(zpe_effects)
        total_loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'ZPE Effects: {zpe_effects}')
    scheduler.step()

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    val_acc = 100 * val_correct / val_total
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc:.2f}%')

# TTA Function
def tta_predict(model, data, num_augmentations=10):
    model.eval()
    outputs = []
    with torch.no_grad():
        outputs.append(model(data))
        aug_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        data_denorm = (data * 0.5) + 0.5
        for _ in range(num_augmentations - 1):
            aug_data = torch.stack([aug_transform(data_denorm[i].cpu()) for i in range(data.size(0))]).to(device)
            output = model(aug_data)
            outputs.append(output)
    return torch.mean(torch.stack(outputs), dim=0)

# Test with TTA
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = tta_predict(model, data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set with TTA: {accuracy:.2f}%')

# Save Model
torch.save(model.state_dict(), '/content/zpe_deepnet_colab.pth')                                                                            
                pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                aug_x = torch.nn.functional.pad(aug_x, [pad_w, pad_w, pad_h, pad_h])
            predictions.append(model(aug_x).unsqueeze(0))
        rotations = [5, -5]
        for angle in rotations:
            theta = torch.tensor([[[torch.cos(torch.tensor(angle * np.pi / 180)), torch.sin(torch.tensor(angle * np.pi / 180)), 0],
                                   [-torch.sin(torch.tensor(angle * np.pi / 180)), torch.cos(torch.tensor(angle * np.pi / 180)), 0]]],
                                 dtype=torch.float32, device=x.device)
            theta = theta.repeat(batch_size, 1, 1)
            grid = torch.nn.functional.affine_grid(theta, [batch_size, x.size(1), h, w], align_corners=True)
            aug_x = torch.nn.functional.grid_sample(x, grid, mode='bilinear', align_corners=True)
            predictions.append(model(aug_x).unsqueeze(0))
        flips = [torch.flip(x, [2]), torch.flip(x, [3])]
        for aug_x in flips:
            predictions.append(model(aug_x).unsqueeze(0))
        weights = torch.tensor([1.0] + [0.75] * 9 + [0.9] * 8 + [0.85] * 8 + [0.8] * 2 + [0.7] * 2, device=x.device, dtype=torch.float32)
        weights = weights / weights.sum()
        weighted_preds = torch.cat(predictions, dim=0) * weights.view(-1, 1, 1)
        return torch.sum(weighted_preds, dim=0)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    model = train_zpe_model()


# Dataset loading and transformations would be implemented here
# based on the user's selected datasets and custom data
`;
    
    const response = await InvokeLLM({
      prompt: `Based on the following complete ZPE PyTorch model template and user-defined parameters, generate the final, complete, and runnable PyTorch training script.

**User Configuration & Hyperparameters:**
${JSON.stringify({project: projectData, params: finalTrainingParams}, null, 2)}

**Instructions:**
1. Use the provided ZPE model templates as base
2. Customize the architecture for task types: ${JSON.stringify(projectData.taskTypes)}
3. Include data loading for datasets: ${JSON.stringify(projectData.selectedDatasets.map(d=>d.identifier))}
4. Generate complete training script with ZPE quantum effects
5. Include proper model save/load functionality
6. Add validation and metrics tracking

Generate the complete, production-ready script.`,
    });
    
    setZpeTrainingCode(response);
    setIsLoading(false);
    setStep(s => s + 1);
  };

  const handleStartTraining = async () => {
    const newProject = await createProject({
      name: projectData.goal.substring(0, 50),
      description: projectData.goal,
      goal: projectData.goal,
      task_types: projectData.taskTypes,
      output_format: projectData.dataFormat || '',
      datasets: projectData.selectedDatasets.map(d => d.identifier),
      constraints: [],
      model_config: finalTrainingParams,
      model_id: '',
      status: 'data_prep',
    });
    const encodedParams = btoa(JSON.stringify(finalTrainingParams));
    window.location.href = `/TrainModel?advisorParams=${encodedParams}&projectId=${newProject.id}`;
    onClose();
  };

  const handleNextStep = () => {
    if (step === 1) {
      fetchDatasetsFromAPI();
      setStep(s => s + 1);
      return;
    } else if (step === 6) { // After Compute & Scale step
      handleGetFinalRecommendations();
    } else if (step === 7) { // After AI Review step
      handleGenerateCodeAndBlueprint();
    } else {
      setStep(s => s + 1);
    }
  };

  const handlePrevStep = () => setStep(s => s - 1);

  const renderStepContent = () => {
    switch(step) {
      case 0:
        return (
          <div>
            <Label className="font-semibold">What is your project's primary goal?</Label>
            <Textarea 
              placeholder="e.g., Build an AI system to generate realistic images of cats, or Create a chatbot for customer support..."
              value={projectData.goal}
              onChange={e => setProjectData({...projectData, goal: e.target.value})}
              className="mt-2 min-h-[100px]"
            />
          </div>
        );

      case 1:
        return (
          <div className="space-y-4">
            <div>
              <Label className="font-semibold text-[#00ffe7] neon-text-glow">What type of AI task(s) do you want to build? (Select all that apply)</Label>
              <div className="mt-4 max-h-[400px] overflow-y-auto pr-2 space-y-4">
                {taskCategories.map((category, index) => (
                  <div key={index} className="space-y-2">
                    <h4 className="font-medium text-sm text-[#ffe600] neon-title-glow border-b border-[#00ffe7] pb-1 sticky top-0 bg-[#0a0f1c] z-10">
                      {category.group}
                    </h4>
                    <div className="grid grid-cols-2 gap-2">
                      {category.tasks.map(task => (
                        <div 
                          key={task.value}
                          onClick={() => handleTaskToggle(task.value)}
                          className={`p-3 rounded-lg border-2 cursor-pointer transition-all flex items-center gap-3 bg-[#10172a] border-[#00ffe7] hover:border-[#ffe600] ${projectData.taskTypes.includes(task.value) ? 'shadow-[0_0_8px_#ffe600]' : ''}`}
                        >
                          <Checkbox 
                            checked={projectData.taskTypes.includes(task.value)}
                            onCheckedChange={() => handleTaskToggle(task.value)}
                            id={task.value}
                            className={`neon-checkbox`}
                          />
                          <Label htmlFor={task.value} className="text-sm font-medium cursor-pointer text-[#fff]">{task.label}</Label>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case 2: // Suggested Datasets
        const recommended = aiSuggestedDatasets.filter(d => d.is_recommended);
        const others = aiSuggestedDatasets.filter(d => !d.is_recommended);

        const DatasetListItem = ({ d, isSelected, onSelect }: { d: any, isSelected: boolean, onSelect: () => void }) => (
          <div
            onClick={onSelect}
            className={`
              flex items-center p-2 rounded-md border-2 cursor-pointer
              transition-all duration-200 ease-in-out
              ${isSelected
                ? 'bg-yellow-500/20 border-yellow-400 neon-glow-yellow-sm' 
                : 'bg-slate-500/10 border-transparent hover:bg-slate-500/20 hover:border-cyan-400'
              }
            `}
          >
            <Checkbox
              checked={isSelected}
              className="mr-3 rounded neon-checkbox"
            />
            <div className="flex-grow">
              <div className="font-semibold text-sm text-cyan-200">{d.name}</div>
              <p className="text-xs text-slate-400 line-clamp-1">{d.description || 'No description available.'}</p>
            </div>
            <a href={d.url} target="_blank" rel="noopener noreferrer" className="p-1 hover:text-yellow-400 transition-colors">
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        );

        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4 -mt-4">
            {/* AI Recommended */}
            <div className="flex flex-col">
              <h4 className="font-semibold text-base text-yellow-400 mb-2">AI Recommended Datasets</h4>
              <div className="space-y-2 pr-2 overflow-y-auto" style={{maxHeight: 'calc(100vh - 350px)'}}>
                {recommended.length > 0 ? (
                  recommended.map(d => (
                    <DatasetListItem
                      key={d.identifier || d.name}
                      d={d}
                      isSelected={projectData.selectedDatasets.some(selected => selected.identifier === d.identifier)}
                      onSelect={() => handleDatasetSelect(d)}
                    />
                  ))
                ) : (
                  <div className="text-center text-sm text-slate-500 py-4">No specific AI recommendations. Check other relevant datasets.</div>
                )}
              </div>
            </div>

            {/* Other Relevant Datasets */}
            <div className="flex flex-col">
              <h4 className="font-semibold text-base text-cyan-400 mb-2">Other Relevant Datasets</h4>
              <div className="space-y-2 pr-2 overflow-y-auto" style={{maxHeight: 'calc(100vh - 350px)'}}>
                {others.length > 0 ? (
                  others.map(d => (
                    <DatasetListItem
                      key={d.identifier || d.name}
                      d={d}
                      isSelected={projectData.selectedDatasets.some(selected => selected.identifier === d.identifier)}
                      onSelect={() => handleDatasetSelect(d)}
                    />
                  ))
                ) : (
                  <div className="text-center text-sm text-slate-500 py-4">No other datasets found.</div>
                )}
              </div>
            </div>
          </div>
        );

      case 3: // Custom Data
        return (
          <div className="space-y-4">
            <Label className="font-semibold">Custom Data Strategy</Label>
            <p className="text-sm text-gray-500">You can also provide your own data or generate synthetic data to complement the suggested datasets.</p>
            <div className="mt-2 space-y-2">
              <div className="flex items-center space-x-2 p-3 bg-gray-50 rounded-lg border">
                <Checkbox 
                  id="data-custom" 
                  checked={projectData.dataStrategies.includes('custom')}
                  onCheckedChange={() => handleDataStrategyToggle('custom')}
                />
                <Label htmlFor="data-custom">Upload my own custom data</Label>
              </div>
              <div className="flex items-center space-x-2 p-3 bg-gray-50 rounded-lg border">
                <Checkbox 
                  id="data-synthesis"
                  checked={projectData.dataStrategies.includes('synthesis')}
                  onCheckedChange={() => handleDataStrategyToggle('synthesis')}
                />
                <Label htmlFor="data-synthesis">Generate synthetic data (AI-created)</Label>
              </div>
            </div>
            
            <AnimatePresence>
              {projectData.dataStrategies.includes('custom') && (
                <motion.div key="custom-data-upload" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="mt-4">
                  <Label>Upload Data File</Label>
                  <Input 
                    type="file" 
                    onChange={async (e) => {
                      const files = e.target.files;
                      if (files && files[0]) {
                        setIsLoading(true);
                        try {
                          const { file_url } = await UploadFile({ file: files[0] });
                          setProjectData(pd => ({...pd, customDataFile: file_url}));
                        } catch (error) {
                          console.error("File upload failed:", error);
                        }
                        setIsLoading(false);
                      }
                    }}
                    className="mt-2" 
                  />
                  {projectData.customDataFile && (
                    <p className="text-sm text-green-600 mt-1">✓ File uploaded successfully</p>
                  )}
                </motion.div>
              )}
              
              {projectData.dataStrategies.includes('synthesis') && (
                <motion.div key="synthesis-data-prompt" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="mt-4">
                  <Label>Data Synthesis Prompt</Label>
                  <Textarea 
                    placeholder="e.g., Generate 1000 product reviews for eco-friendly water bottles, focusing on sustainability, durability, and design..."
                    value={projectData.dataSynthesisPrompt}
                    onChange={e => setProjectData({...projectData, dataSynthesisPrompt: e.target.value})}
                    className="mt-2"
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );

      case 4: // Task Specifics
        return (
          <div className="space-y-4">
            <Label className="font-semibold">Task-Specific Configuration</Label>
            <div className="max-h-[400px] overflow-y-auto pr-2 space-y-4">
              <AnimatePresence>
                {projectData.taskTypes.some(t => t.includes('classification')) && (
                  <motion.div key="classification-settings" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="space-y-3">
                    <h4 className="font-medium text-sm text-gray-700 pt-2">Classification Settings</h4>
                    <div>
                      <Label>Number of categories/classes</Label>
                      <Input 
                        type="number" 
                        placeholder="e.g., 10 for digit classification"
                        value={projectData.numClasses}
                        onChange={e => setProjectData({...projectData, numClasses: e.target.value})}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Target accuracy (%)</Label>
                      <Input 
                        type="number" 
                        placeholder="e.g., 95"
                        value={projectData.accuracyTarget}
                        onChange={e => setProjectData({...projectData, accuracyTarget: e.target.value})}
                        className="mt-1"
                      />
                    </div>
                  </motion.div>
                )}

                {projectData.taskTypes.some(t => t.includes('generation')) && (
                  <motion.div key="generation-settings" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="space-y-3">
                    <h4 className="font-medium text-sm text-gray-700 pt-2">Generation Settings</h4>
                    {projectData.taskTypes.some(t => t.includes('text')) && (
                      <div>
                        <Label>Desired text style/tone</Label>
                        <Input 
                          placeholder="e.g., formal, casual, creative, technical"
                          value={projectData.textStyle}
                          onChange={e => setProjectData({...projectData, textStyle: e.target.value})}
                          className="mt-1"
                        />
                      </div>
                    )}
                    {projectData.taskTypes.some(t => t.includes('image')) && (
                      <div>
                        <Label>Target image resolution</Label>
                        <Select 
                          value={projectData.imageResolution}
                          onValueChange={v => setProjectData({...projectData, imageResolution: v})}
                        >
                          <SelectTrigger className="mt-1">
                            <SelectValue placeholder="Select resolution..." />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="256x256">256x256 (Fast)</SelectItem>
                            <SelectItem value="512x512">512x512 (Balanced)</SelectItem>
                            <SelectItem value="1024x1024">1024x1024 (High Quality)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                  </motion.div>
                )}

                {projectData.taskTypes.some(t => t.includes('detection')) && (
                  <motion.div key="detection-settings" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="space-y-3">
                    <h4 className="font-medium text-sm text-gray-700 pt-2">Detection Settings</h4>
                    <div>
                      <Label>Average object size to detect</Label>
                      <Select 
                        value={projectData.objectSize}
                        onValueChange={v => setProjectData({...projectData, objectSize: v})}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Select object size..." />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="small">Small objects (coins, screws)</SelectItem>
                          <SelectItem value="medium">Medium objects (faces, cars)</SelectItem>
                          <SelectItem value="large">Large objects (buildings, landscapes)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label>Real-time processing required?</Label>
                      <Select 
                        value={projectData.realTimeRequirement}
                        onValueChange={v => setProjectData({...projectData, realTimeRequirement: v})}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Select requirement..." />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes (live video/camera)</SelectItem>
                          <SelectItem value="no">No (batch processing)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </motion.div>
                )}

                {projectData.taskTypes.some(t => t.includes('speech') || t.includes('audio') || t.includes('music')) && (
                  <motion.div key="audio-settings" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="space-y-3">
                    <h4 className="font-medium text-sm text-gray-700 pt-2">Audio Settings</h4>
                    <div>
                      <Label>Audio length/duration</Label>
                      <Select 
                        value={projectData.audioLength}
                        onValueChange={v => setProjectData({...projectData, audioLength: v})}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Select duration..." />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="short">Short (less than 10 seconds)</SelectItem>
                          <SelectItem value="medium">Medium (10 seconds - 2 minutes)</SelectItem>
                          <SelectItem value="long">Long (more than 2 minutes)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </motion.div>
                )}

                <div>
                  <Label>Primary data format</Label>
                  <Select 
                    value={projectData.dataFormat}
                    onValueChange={v => setProjectData({...projectData, dataFormat: v})}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select data format..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="images">Images (JPG, PNG, etc.)</SelectItem>
                      <SelectItem value="text">Text (documents, messages)</SelectItem>
                      <SelectItem value="audio">Audio (MP3, WAV, etc.)</SelectItem>
                      <SelectItem value="video">Video (MP4, AVI, etc.)</SelectItem>
                      <SelectItem value="tabular">Tabular (CSV, Excel)</SelectItem>
                      <SelectItem value="time_series">Time Series Data</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </AnimatePresence>
            </div>
          </div>
        );

      case 5: // Model Strategy
        return (
          <div className="space-y-4">
            <Label className="font-semibold">Model Strategy</Label>
            <RadioGroup 
              value={projectData.modelStrategy} 
              onValueChange={v => setProjectData({...projectData, modelStrategy: v})}
              className="mt-2"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="scratch" id="model-scratch" />
                <Label htmlFor="model-scratch">Train from scratch with ZPE quantum enhancement</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="merge" id="model-merge" />
                <Label htmlFor="model-merge">Fine-tune existing open-source model with ZPE</Label>
              </div>
            </RadioGroup>
            
            {projectData.modelStrategy === 'merge' && (
              <motion.div initial={{opacity:0}} animate={{opacity:1}} className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800 flex items-center gap-2">
                <Lightbulb className="w-4 h-4 shrink-0"/>
                The AI will suggest the most suitable open-source model for your specific task and merge it with ZPE quantum enhancements.
              </motion.div>
            )}
          </div>
        );

      case 6: // Compute & Scale
        return (
          <div className="space-y-4">
            <div>
              <Label className="font-semibold">Compute Choice</Label>
              <RadioGroup 
                value={projectData.computeChoice} 
                onValueChange={v => setProjectData({...projectData, computeChoice: v})}
                className="mt-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="cloud" id="compute-cloud" />
                  <Label htmlFor="compute-cloud">Cloud GPU (Recommended - Auto-scaling)</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="on-prem" id="compute-on-prem" />
                  <Label htmlFor="compute-on-prem">Local GPU (Your hardware)</Label>
                </div>
              </RadioGroup>
            </div>
            <div>
              <Label className="font-semibold">Expected Production Scale (users per month)</Label>
              <Select 
                value={projectData.productionScale}
                onValueChange={v => setProjectData({...projectData, productionScale: v})}
              >
                <SelectTrigger className="mt-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 user (Personal/Testing)</SelectItem>
                  <SelectItem value="50">50 users (Small Team)</SelectItem>
                  <SelectItem value="500">500 users (Growing Business)</SelectItem>
                  <SelectItem value="5000">5,000 users (Medium Business)</SelectItem>
                  <SelectItem value="50000">50,000+ users (Large Scale)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        );
      
      case 7: // AI Review
        return recommendations && (
          <div className="space-y-4">
            <h4 className="font-semibold text-lg">AI Recommendations</h4>
            <p className="text-sm text-gray-500">The AI has generated the final blueprint for your model based on your selections.</p>
            
            {recommendations.architecture && (
              <div>
                <h5 className="font-semibold mt-4 mb-2">Recommended Architecture</h5>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <p className="font-medium text-gray-900">{recommendations.architecture.recommended}</p>
                  <p className="text-sm text-gray-600 mt-1">{recommendations.architecture.reasoning}</p>
                </div>
              </div>
            )}

            {recommendations.gpu_recommendation && (
                <div>
                  <h5 className="font-semibold mt-4 mb-2">Recommended GPU</h5>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="font-medium text-gray-900">{recommendations.gpu_recommendation}</p>
                  </div>
                </div>
              )}
          </div>
        );

      case 8: // AI Blueprint
        return recommendations && (
          <div className="space-y-4">
            <h4 className="font-semibold text-lg">Your AI Blueprint is Ready</h4>
            
            <div className="p-4 bg-gray-50 rounded-lg space-y-3">
              <div>
                <h5 className="font-semibold text-gray-800">Task:</h5>
                <p className="text-sm text-purple-700">
                  {projectData.taskTypes.map(t => t.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())).join(', ')}
                </p>
              </div>
              
              <div>
                <h5 className="font-semibold text-gray-800">Selected Datasets:</h5>
                <p className="text-sm text-purple-700">
                  {projectData.selectedDatasets.map(d => d.name).join(', ') || 'None selected'}
                </p>
              </div>
              
              <div>
                <h5 className="font-semibold text-gray-800">Model Architecture:</h5>
                <p className="text-sm">{recommendations.architecture?.recommended}</p>
              </div>
              
              <div>
                <h5 className="font-semibold text-gray-800">Recommended GPU:</h5>
                <p className="text-sm">{recommendations.gpu_recommendation}</p>
              </div>
            </div>
            
            <Button 
              onClick={() => setIsCodeModalOpen(true)} 
              variant="outline" 
              className="w-full flex items-center gap-2"
            >
              <Eye className="w-4 h-4" />
              View Training Script
            </Button>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <CodeModal 
        isOpen={isCodeModalOpen} 
        onClose={() => setIsCodeModalOpen(false)} 
        code={zpeTrainingCode} 
      />
      <style>{`
        .neon-glow {
          box-shadow: 0 0 32px #00ffe7, 0 0 8px #a259ff;
        }
        .neon-text-glow {
          text-shadow: 0 0 8px #a259ff, 0 0 2px #fff;
        }
        .neon-btn-glow {
          box-shadow: 0 0 8px #00ffe7, 0 0 2px #a259ff;
        }
        .neon-progress {
          background: linear-gradient(90deg, #00ffe7 0%, #ffe600 100%);
          box-shadow: 0 0 16px #00ffe7, 0 0 8px #ffe600;
          height: 10px;
          border-radius: 6px;
          animation: neon-pulse 2s infinite alternate;
        }
        @keyframes neon-pulse {
          0% { box-shadow: 0 0 32px #00ffe7, 0 0 8px #a259ff; }
          100% { box-shadow: 0 0 48px #ffe600, 0 0 16px #a259ff; }
        }
        .holo-heading {
          background: linear-gradient(90deg, #00ffe7 0%, #a259ff 50%, #ffe600 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          text-shadow: 0 0 12px #a259ff, 0 0 2px #fff;
        }
        .neon-input {
          background: #10172a;
          border: 2px solid #00ffe7;
          color: #fff;
          box-shadow: 0 0 8px #00ffe7;
        }
        .neon-input:focus {
          border-color: #a259ff;
          box-shadow: 0 0 16px #a259ff;
        }
        .neon-title-glow {
          color: #ffe600;
          text-shadow: 0 0 16px #ffe600, 0 0 4px #fff;
          font-weight: 900;
          letter-spacing: 0.04em;
        }
        .neon-checkbox {
          transition: box-shadow 0.2s, border-color 0.2s, background 0.2s;
        }
        .neon-checkbox[data-state="checked"] {
          background: rgba(255,230,0,0.35);
          border-color: #ffe600 !important;
          box-shadow: 0 0 12px #ffe600, 0 0 4px #fffbe6;
          backdrop-filter: blur(2px) saturate(1.5);
        }
        .neon-checkbox[data-state="unchecked"] {
          background: transparent;
          border-color: #00ffe7;
        }
      `}</style>
      <Card className="border-2 border-[#00ffe7] shadow-[0_0_32px_#00ffe7] bg-[#0a0f1c] w-full max-w-4xl neon-glow">
        <CardHeader className="border-b-2 border-[#00ffe7] bg-[#0a0f1c] neon-glow">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 neon-title-glow">
              <Brain className="w-6 h-6 text-[#a259ff] neon-text-glow" />
              <span>AI Model Builder</span>
            </CardTitle>
            <Button variant="ghost" size="icon" onClick={onClose} className="text-[#00ffe7] hover:bg-[#10172a]">
              <X className="w-5 h-5" />
            </Button>
          </div>
          <div className="flex items-center gap-1 mt-4 overflow-x-auto pb-2">
            {steps.map((s, index) => (
              <React.Fragment key={index}>
                <div className={`flex items-center gap-2 shrink-0 ${index <= step ? 'text-[#a259ff] neon-text-glow' : 'text-[#00ffe7]/40'}`}> 
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${index < step ? 'bg-[#a259ff]/30 border-2 border-[#a259ff] neon-glow animate-pulse' : 'bg-[#1a2233] border border-[#00ffe7]/30'}`}>
                    {index < step ? (
                      <CheckCircle className="w-5 h-5 text-[#a259ff] neon-text-glow" />
                    ) : (
                      s.icon && <s.icon className="w-5 h-5 text-[#00ffe7]" />
                    )}
                  </div>
                  <span className="text-sm font-medium">{s.title}</span>
                </div>
                {index < steps.length - 1 && (
                  <div className="w-6 h-1 rounded-full neon-progress mx-1 shrink-0" />
                )}
              </React.Fragment>
            ))}
          </div>
        </CardHeader>
        <CardContent className="p-6 min-h-[300px] bg-[#10172a] neon-glow">
          <AnimatePresence mode="wait">
            <motion.div 
              key={step} 
              initial={{ opacity: 0, x: 20 }} 
              animate={{ opacity: 1, x: 0 }} 
              exit={{ opacity: 0, x: -20 }} 
              transition={{ duration: 0.2 }}
            >
              {isLoading && step !== 2 ? (
                <div className="flex flex-col justify-center items-center h-full min-h-[300px]">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  >
                    <Brain className="w-12 h-12 text-[#a259ff] neon-text-glow animate-pulse" />
                  </motion.div>
                  <p className="mt-4 text-[#a259ff] neon-text-glow">AI is thinking...</p>
                </div>
              ) : (
                renderStepContent()
              )}
            </motion.div>
          </AnimatePresence>
        </CardContent>
        <div className="flex justify-between items-center mt-6 p-4 border-t-2 border-[#00ffe7] bg-[#0a0f1c] rounded-b-lg neon-glow">
          <Button 
            variant="outline" 
            onClick={step > 0 ? handlePrevStep : onClose}
            disabled={isLoading}
            className="neon-btn-glow border-[#a259ff] text-[#a259ff] hover:bg-[#10172a]"
          >
            {step === 0 ? "Cancel" : "Back"}
          </Button>
          <div className="flex gap-2">
            {step < 8 && (
              <Button 
                onClick={handleNextStep}
                disabled={
                  (step === 0 && !projectData.goal) ||
                  (step === 1 && projectData.taskTypes.length === 0) ||
                  (step === 2 && projectData.selectedDatasets.length === 0) ||
                  isLoading
                }
                className="neon-btn-glow bg-[#a259ff] text-black hover:bg-[#00ffe7] hover:text-black border-2 border-[#a259ff] animate-pulse"
              >
                {step === 6 ? "Get AI Recommendations" : "Next"}
                {isLoading && (step === 1 || step === 6) && <Brain className="w-4 h-4 ml-2 animate-pulse text-[#a259ff]" />}
              </Button>
            )}
            {step === 8 && (
              <Button 
                onClick={handleStartTraining} 
                disabled={isLoading}
                className="neon-btn-glow bg-[#39ff14] hover:bg-[#00ffe7] text-black border-2 border-[#39ff14] animate-pulse"
              >
                <Play className="w-4 h-4 mr-2" />
                Start Training Your AI
              </Button>
            )}
          </div>
        </div>
      </div>
    );
}