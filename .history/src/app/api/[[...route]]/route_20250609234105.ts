// src/app/api/[[...route]]/route.ts
import { genkit } from "genkit";
import { googleAI } from "@genkit-ai/googleai";
import {NextRequest} from 'next/server';

import "@/ai/flows/adapt-zpe";
import "@/ai/flows/approximate-zpe";
import "@/ai/flows/configure-model-for-dataset";
import "@/ai/flows/get-quantum-explanation-flow";
import "@/ai/flows/extract-high-gain-components";
import "@/ai/flows/get-initial-zpe-analysis-flow";
import "@/ai/flows/hs-qnn-parameter-advisor";
import "@/ai/flows/implement-zpe";
import "@/ai/flows/invoke-generic-llm-flow";
import "@/ai/flows/quantize-colab-model";


genkit({
  plugins: [
    googleAI(),
  ],
});

export async function POST(req: NextRequest) {
  return genkit(req as any);
}
