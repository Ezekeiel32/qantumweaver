"use client";
import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Database, 
  Upload, 
  Plus, 
  FileText, 
  Image, 
  BarChart3, 
  Download, 
  Trash2, 
  Eye,
  Search,
  Filter,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Clock,
  HardDrive,
  Cloud,
  Zap
} from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { motion, AnimatePresence } from "framer-motion";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

interface Dataset {
  id: string;
  name: string;
  type: string;
  size: string;
  uploadDate: string;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  features?: number;
  samples?: number;
  description?: string;
  tags?: string[];
}

export default function DataPortal() {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all");
  const { toast } = useToast();

  // Mock datasets for demonstration
  useEffect(() => {
    const mockDatasets: Dataset[] = [
      {
        id: "1",
        name: "MNIST Quantum Enhanced",
        type: "image",
        size: "12.5 MB",
        uploadDate: "2024-01-15",
        status: "ready",
        features: 784,
        samples: 70000,
        description: "Handwritten digits dataset with quantum noise augmentation",
        tags: ["classification", "images", "quantum-enhanced"]
      },
      {
        id: "2",
        name: "CIFAR-10 ZPE",
        type: "image",
        size: "170.2 MB",
        uploadDate: "2024-01-10",
        status: "ready",
        features: 3072,
        samples: 60000,
        description: "Color image classification with ZPE field analysis",
        tags: ["classification", "color-images", "zpe-analysis"]
      },
      {
        id: "3",
        name: "Quantum Sensor Data",
        type: "tabular",
        size: "45.8 MB",
        uploadDate: "2024-01-08",
        status: "ready",
        features: 128,
        samples: 50000,
        description: "Time-series data from quantum sensors",
        tags: ["regression", "time-series", "quantum-sensors"]
      },
      {
        id: "4",
        name: "ZPE Particle Trajectories",
        type: "tabular",
        size: "89.3 MB",
        uploadDate: "2024-01-05",
        status: "processing",
        features: 256,
        samples: 100000,
        description: "Particle movement data under ZPE influence",
        tags: ["physics", "particles", "zpe-effects"]
      }
    ];
    
    setTimeout(() => {
      setDatasets(mockDatasets);
      setIsLoading(false);
    }, 1000);
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      toast({ 
        title: "No file selected", 
        description: "Please choose a file to upload.", 
        variant: "destructive" 
      });
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append("file", selectedFile);
    
    // Simulate progress for better UX
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.ok) {
        const result = await response.json();
        
        // Add new dataset to list
        const newDataset: Dataset = {
          id: result.id || Date.now().toString(),
          name: selectedFile.name,
          type: selectedFile.type.startsWith('image/') ? 'image' : 'tabular',
          size: formatFileSize(selectedFile.size),
          uploadDate: new Date().toISOString().split('T')[0],
          status: 'processing',
          features: Math.floor(Math.random() * 500) + 50,
          samples: Math.floor(Math.random() * 10000) + 1000,
          description: `Uploaded ${selectedFile.name}`,
          tags: [selectedFile.type.startsWith('image/') ? 'images' : 'tabular']
        };
        
        setDatasets(prev => [newDataset, ...prev]);
        
        toast({ 
          title: "Upload Successful", 
          description: `File ${result.filename} uploaded successfully.` 
        });
        
        // Simulate processing completion
        setTimeout(() => {
          setDatasets(prev => prev.map(ds => 
            ds.id === newDataset.id ? { ...ds, status: 'ready' as const } : ds
          ));
        }, 3000);
        
      } else {
        const errorData = await response.json();
        toast({ 
          title: "Upload Failed", 
          description: errorData.detail || "Failed to upload file", 
          variant: "destructive" 
        });
      }
    } catch (error) {
      clearInterval(progressInterval);
      toast({ 
        title: "Connection Error", 
        description: "Could not upload file. Is the backend running?", 
        variant: "destructive" 
      });
    } finally {
      setTimeout(() => {
        setIsUploading(false);
        setSelectedFile(null);
        setUploadProgress(0);
      }, 1000);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: Dataset['status']) => {
    switch (status) {
      case 'uploading':
        return <RefreshCw className="w-4 h-4 animate-spin text-blue-500" />;
      case 'processing':
        return <Clock className="w-4 h-4 text-orange-500" />;
      case 'ready':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: Dataset['status']) => {
    switch (status) {
      case 'uploading':
        return 'bg-blue-100 text-blue-800';
      case 'processing':
        return 'bg-orange-100 text-orange-800';
      case 'ready':
        return 'bg-green-100 text-green-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const filteredDatasets = datasets.filter(dataset => {
    const matchesSearch = dataset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         dataset.description?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === "all" || dataset.type === filterType;
    return matchesSearch && matchesFilter;
  });

  const handleDeleteDataset = async (datasetId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        setDatasets(prev => prev.filter(ds => ds.id !== datasetId));
        toast({ 
          title: "Dataset Deleted", 
          description: "Dataset has been removed successfully." 
        });
      } else {
        toast({ 
          title: "Delete Failed", 
          description: "Could not delete dataset.", 
          variant: "destructive" 
        });
      }
    } catch (error) {
      toast({ 
        title: "Error", 
        description: "Failed to delete dataset.", 
        variant: "destructive" 
      });
    }
  };

  return (
    <div className="container mx-auto p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-2 md:space-y-0">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-primary flex items-center gap-2">
            <Database className="h-8 w-8" /> Data Portal
          </h1>
          <p className="text-muted-foreground">Manage datasets for quantum-enhanced machine learning</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Upload Section */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5 text-blue-600" />
            Upload New Dataset
          </CardTitle>
          <CardDescription>
            Upload your dataset for quantum-enhanced analysis and training
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="border-2 border-dashed border-blue-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
            <input
              type="file"
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              accept=".csv,.json,.txt,.png,.jpg,.jpeg,.xlsx,.parquet"
            />
            <Upload className="w-12 h-12 text-blue-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-slate-800 mb-2">
              Drop your dataset here
            </h3>
            <p className="text-slate-600 mb-4">
              Supports CSV, JSON, images, and more. Quantum analysis will be applied automatically.
            </p>
            {selectedFile && (
              <div className="bg-blue-50 p-3 rounded-lg">
                <p className="text-sm text-blue-800">
                  Selected: {selectedFile.name} ({formatFileSize(selectedFile.size)})
                </p>
              </div>
            )}
          </div>
          
          {isUploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Uploading and analyzing...</span>
                <span>{Math.round(uploadProgress)}%</span>
              </div>
              <Progress value={uploadProgress} />
              <div className="bg-blue-50 p-3 rounded-lg">
                <p className="text-sm text-blue-800">
                  🔬 Quantum analysis in progress...
                </p>
              </div>
            </div>
          )}
          
          <Button 
            onClick={handleFileUpload} 
            disabled={isUploading || !selectedFile}
            className="w-full bg-blue-600 hover:bg-blue-700"
          >
            <Upload className="h-4 w-4 mr-2" />
            {isUploading ? 'Uploading...' : 'Upload Dataset'}
          </Button>
        </CardContent>
      </Card>

      {/* Datasets List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            Your Datasets
          </CardTitle>
          <CardDescription>
            Manage and explore your uploaded datasets
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Search and Filter */}
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search datasets..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <div className="flex gap-2">
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md bg-white"
              >
                <option value="all">All Types</option>
                <option value="image">Images</option>
                <option value="tabular">Tabular</option>
              </select>
            </div>
          </div>

          {/* Datasets Grid */}
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 animate-spin text-blue-500 mr-2" />
              <span>Loading datasets...</span>
            </div>
          ) : filteredDatasets.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <Database className="w-12 h-12 mx-auto mb-4 text-gray-300" />
              <p>No datasets found</p>
              <p className="text-sm">Upload your first dataset to get started</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <AnimatePresence>
                {filteredDatasets.map((dataset) => (
                  <motion.div
                    key={dataset.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Card className="hover:shadow-lg transition-all duration-200 cursor-pointer">
                      <CardHeader className="pb-3">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-2">
                            {dataset.type === 'image' ? (
                              <Image className="w-5 h-5 text-blue-500" />
                            ) : (
                              <BarChart3 className="w-5 h-5 text-green-500" />
                            )}
                            <div>
                              <CardTitle className="text-lg">{dataset.name}</CardTitle>
                              <div className="flex items-center gap-2 mt-1">
                                {getStatusIcon(dataset.status)}
                                <Badge className={getStatusColor(dataset.status)}>
                                  {dataset.status}
                                </Badge>
                              </div>
                            </div>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-gray-600">Size:</span>
                            <p className="font-medium">{dataset.size}</p>
                          </div>
                          <div>
                            <span className="text-gray-600">Type:</span>
                            <p className="font-medium capitalize">{dataset.type}</p>
                          </div>
                          {dataset.features && (
                            <div>
                              <span className="text-gray-600">Features:</span>
                              <p className="font-medium">{dataset.features.toLocaleString()}</p>
                            </div>
                          )}
                          {dataset.samples && (
                            <div>
                              <span className="text-gray-600">Samples:</span>
                              <p className="font-medium">{dataset.samples.toLocaleString()}</p>
                            </div>
                          )}
                        </div>
                        
                        {dataset.description && (
                          <p className="text-sm text-gray-600 line-clamp-2">
                            {dataset.description}
                          </p>
                        )}
                        
                        {dataset.tags && dataset.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {dataset.tags.slice(0, 3).map((tag, index) => (
                              <Badge key={index} variant="outline" className="text-xs">
                                {tag}
                              </Badge>
                            ))}
                            {dataset.tags.length > 3 && (
                              <Badge variant="outline" className="text-xs">
                                +{dataset.tags.length - 3}
                              </Badge>
                            )}
                          </div>
                        )}
                        
                        <div className="flex items-center justify-between pt-2 border-t">
                          <span className="text-xs text-gray-500">
                            {dataset.uploadDate}
                          </span>
                          <div className="flex gap-1">
                            <Button size="sm" variant="outline">
                              <Eye className="w-3 h-3" />
                            </Button>
                            <Button size="sm" variant="outline">
                              <Download className="w-3 h-3" />
                            </Button>
                            <Button 
                              size="sm" 
                              variant="outline"
                              onClick={() => handleDeleteDataset(dataset.id)}
                            >
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-r from-blue-50 to-blue-100">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-600" />
              <div>
                <p className="text-sm text-blue-600">Total Datasets</p>
                <p className="text-2xl font-bold text-blue-800">{datasets.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gradient-to-r from-green-50 to-green-100">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <HardDrive className="w-5 h-5 text-green-600" />
              <div>
                <p className="text-sm text-green-600">Total Size</p>
                <p className="text-2xl font-bold text-green-800">
                  {datasets.reduce((acc, ds) => {
                    const size = parseFloat(ds.size.split(' ')[0]);
                    const unit = ds.size.split(' ')[1];
                    return acc + (unit === 'GB' ? size * 1024 : unit === 'MB' ? size : size / 1024);
                  }, 0).toFixed(1)} MB
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gradient-to-r from-purple-50 to-purple-100">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-purple-600" />
              <div>
                <p className="text-sm text-purple-600">Ready</p>
                <p className="text-2xl font-bold text-purple-800">
                  {datasets.filter(ds => ds.status === 'ready').length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gradient-to-r from-orange-50 to-orange-100">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Cloud className="w-5 h-5 text-orange-600" />
              <div>
                <p className="text-sm text-orange-600">Processing</p>
                <p className="text-2xl font-bold text-orange-800">
                  {datasets.filter(ds => ds.status === 'processing').length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 