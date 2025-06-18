import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Database, Upload, Plus } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

const API_BASE_URL = "http://localhost:8000";

// (Full DataPortal code from user message)
// ... existing code ... 