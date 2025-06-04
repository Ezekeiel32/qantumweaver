// src/app/api/[[...route]]/route.ts
import { genkit } from "genkit";
import { googleAI } from "@genkit-ai/googleai";
import {NextRequest} from 'next/server';

import "@/ai/flows/adapt-zpe";
import "@/ai/flows/approximate-zpe";
import "@/ai/flows/configure-model-for-dataset";
import "@/ai/flows/extract-high-gain-components";
import "@/ai/flows/get-initial-zpe-analysis-flow";
import "@/ai/flows/get-quantum-explanation-flow";
import "@/ai/flows/get-zpe-chat-response-flow";
import "@/ai/flows/hs-qnn-parameter-advisor";
import "@/ai/flows/implement-zpe";
import "@/ai/flows/invoke-generic-llm-flow";
import "@/ai/flows/quantize-colab-model";
import "@/ai/flows/show-scaled-output";


genkit({
  plugins: [
    googleAI(),
  ],
  logLevel: "debug",
  enableTracingAndMetrics: true,
});

export async function POST(req: NextRequest) {
  const G = await import('genkit/next');
  return G.POST(req)
}
