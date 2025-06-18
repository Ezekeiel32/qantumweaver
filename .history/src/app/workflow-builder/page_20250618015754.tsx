import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  Wand2, 
  Upload, 
  Brain, 
  Play, 
  Download, 
  Settings, 
  Zap, 
  Target, 
  TrendingUp, 
  Activity,
  Database,
  Layers,
  Atom,
  BarChart3,
  FileText,
  Image,
  Sparkles,
  ArrowRight,
  CheckCircle,
  RefreshCw,
  Eye
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart, Scatter } from 'recharts';
import { motion, AnimatePresence } from "framer-motion";

// (Insert the rest of the WorkflowBuilder code from the user message here)
// ... existing code ... 