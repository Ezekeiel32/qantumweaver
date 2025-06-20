// src/app/api/[[...route]]/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { TrainingParameters } from '@/types/training';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import fs from 'fs';

// Helper function to validate training parameters
function validateTrainingParams(params: any): params is TrainingParameters {
  const requiredFields = [
    'totalEpochs',
    'batchSize',
    'learningRate',
    'weightDecay',
    'momentumParams',
    'strengthParams',
    'noiseParams',
    'couplingParams',
    'quantumCircuitSize',
    'labelSmoothing',
    'quantumMode',
    'modelName'
  ];

  return requiredFields.every(field => field in params);
}

// Helper function to extract job ID from URL
function extractJobId(url: string): string | null {
  const match = url.match(/\/api\/(status|stop)\/([^\/]+)$/);
  return match ? match[2] : null;
}

// Training endpoint handler
export async function POST(req: NextRequest) {
  try {
    const data = await req.json();
    
    if (!validateTrainingParams(data)) {
      return NextResponse.json({ error: 'Invalid training parameters' }, { status: 400 });
    }

    const jobId = `zpe_job_${uuidv4().split('-')[0]}`;
    const jobData = {
      job_id: jobId,
      status: 'pending',
      parameters: data,
      start_time: new Date().toISOString(),
      log_messages: [],
      metrics: []
    };

    // Save job data
    const jobsDir = path.join(process.cwd(), 'logs.json');
    if (!fs.existsSync(jobsDir)) {
      fs.mkdirSync(jobsDir, { recursive: true });
    }
    fs.writeFileSync(path.join(jobsDir, `${jobId}.json`), JSON.stringify(jobData, null, 2));

    // Start training process
    const pythonProcess = spawn('python', [
      'train.py',
      '--job-id', jobId,
      '--params', JSON.stringify(data)
    ]);

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Training error: ${data}`);
    });

    return NextResponse.json({ job_id: jobId });
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Status endpoint handler
export async function GET(req: NextRequest) {
  try {
    const jobId = extractJobId(req.url);
    if (!jobId) {
      return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
    }

    const jobFile = path.join(process.cwd(), 'logs.json', `${jobId}.json`);
    if (!fs.existsSync(jobFile)) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const jobData = JSON.parse(fs.readFileSync(jobFile, 'utf-8'));
    return NextResponse.json(jobData);
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Stop endpoint handler
export async function PUT(req: NextRequest) {
  try {
    const jobId = extractJobId(req.url);
    if (!jobId) {
      return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
    }

    const jobFile = path.join(process.cwd(), 'logs.json', `${jobId}.json`);
    if (!fs.existsSync(jobFile)) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const jobData = JSON.parse(fs.readFileSync(jobFile, 'utf-8'));
    jobData.status = 'stopped';
    fs.writeFileSync(jobFile, JSON.stringify(jobData, null, 2));

    return NextResponse.json({ status: 'stopped' });
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
