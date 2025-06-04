'use server';
/**
 * @fileOverview An AI agent to advise on parameters for the next step in a Hilbert Space Quantum Neural Network (HS-QNN) sequence.
 *
 * - adviseHSQNNParameters - A function that analyzes a previous job's ZPE state and an HNN objective to suggest new training parameters.
 * - HSQNNAdvisorInput - The input type for the adviseHSQNNParameters function.
 * - HSQNNAdvisorOutput - The return type for the adviseHSQNNParameters function.
 */

import {z} from 'genkit';
import {ai} from '../genkit';

// Define the Zod schema for TrainingParameters locally for the flow if not easily importable
const TrainingParametersSchemaInternal = z.object({
  totalEpochs: z.number().int().min(1).max(200),
  batchSize: z.number().int().min(8).max(256),
  learningRate: z.number().min(0.00001).max(0.1),
  weightDecay: z.number().min(0).max(0.1),
  momentumParams: z.array(z.number().min(0).max(1)).length(6, "Momentum parameters must have 6 values."),
  strengthParams: z.array(z.number().min(0).max(1)).length(6, "Strength parameters must have 6 values."),
  noiseParams: z.array(z.number().min(0).max(1)).length(6, "Noise parameters must have 6 values."),
  couplingParams: z.array(z.number().min(0).max(1)).length(6, "Coupling parameters must have 6 values."),
  quantumCircuitSize: z.number().int().min(4).max(64),
  labelSmoothing: z.number().min(0).max(0.5),
  quantumMode: z.boolean(),
  modelName: z.string().min(1),
  baseConfigId: z.string().nullable().optional(), // Allow null
});

const HSQNNAdvisorInputSchema = z.object({
  previousJobId: z.string().describe("The ID of the completed job being analyzed."),
  previousZpeEffects: z.array(z.number()).length(6).describe("The final ZPE effects (array of 6 numbers, e.g., [0.199, 0.011, ...]) from the previous job. These values typically range from 0.0 to 0.2, but can vary."),
  previousTrainingParameters: TrainingParametersSchemaInternal.describe("The full set of training parameters used for the previous job."),
  hnnObjective: z.string().min(10).describe("The user's objective for the next HNN training step. E.g., 'Maximize accuracy while keeping ZPE effects in the 0.05-0.15 range', 'Aggressively explore higher ZPE magnitudes for layer 3', 'Stabilize overall ZPE variance and slightly increase learning rate if accuracy was high'.")
});
export type HSQNNAdvisorInput = z.infer<typeof HSQNNAdvisorInputSchema>;

const HSQNNAdvisorOutputSchema = z.object({
  suggestedNextTrainingParameters: z.object({
    momentumParams: z.array(z.number().min(0).max(1)).length(6, "Momentum parameters must have 6 values."),
    strengthParams: z.array(z.number().min(0).max(1)).length(6, "Strength parameters must have 6 values."),
    noiseParams: z.array(z.number().min(0).max(1)).length(6, "Noise parameters must have 6 values."),
    couplingParams: z.array(z.number().min(0).max(1)).length(6, "Coupling parameters must have 6 values."),
    totalEpochs: z.number().int().min(1).max(200).optional(),
    batchSize: z.number().int().min(8).max(256).optional(),
    learningRate: z.number().min(0.00001).max(0.1).optional(),
    weightDecay: z.number().min(0).max(0.1).optional(),
    quantumCircuitSize: z.number().int().min(4).max(64).optional(),
    labelSmoothing: z.number().min(0).max(0.5).optional(),
    quantumMode: z.boolean().optional(),
    modelName: z.string().min(1).optional(),
    baseConfigId: z.string().nullable().optional(),
  }).describe("Suggested training parameters for the next HNN job. This object will contain all relevant parameters."),
  reasoning: z.string().describe("A step-by-step explanation of why these parameters are suggested, linking back to the previous ZPE state, parameters, and the HNN objective. Should mention which ZPE values (high/low/average) influenced decisions.")
});
export type HSQNNAdvisorOutput = z.infer<typeof HSQNNAdvisorOutputSchema>;

export async function adviseHSQNNParameters(input: HSQNNAdvisorInput): Promise<HSQNNAdvisorOutput> {
  return hsQnnParameterAdvisorFlow(input);
}

const prompt = ai.definePrompt({
  name: 'hsQnnParameterAdvisorPrompt',
  output: {schema: HSQNNAdvisorOutputSchema},
  prompt: `You are an expert AI Quantum Neural Network Engineer Assistant specializing in Zero-Point Energy (ZPE) enhanced Quantum Neural Networks and their sequential training in a Hilbert Space Quantum Neural Network (HS-QNN) framework.

The user has completed a training job and wants advice on parameters for the *next* job in an HNN sequence.
Your goal is to analyze the ZPE effects and parameters of the previous job, consider the user\'s stated HNN objective, and suggest a new set of training parameters.

Previous Job Details:
- Job ID: {{{previousJobId}}}
- Final ZPE Effects (6 layers): {{{previousZpeEffects}}} (These values typically range from 0.0 to 0.2, indicating the average deviation of ZPE flow from 1.0 for each layer. Higher values mean stronger ZPE effect/perturbation for that layer.)
- Training Parameters Used:
  - Model Name: {{{previousTrainingParameters.modelName}}}
  - Total Epochs: {{{previousTrainingParameters.totalEpochs}}}
  - Batch Size: {{{previousTrainingParameters.batchSize}}}
  - Learning Rate: {{{previousTrainingParameters.learningRate}}}
  - Weight Decay: {{{previousTrainingParameters.weightDecay}}}
  - Momentum Params (ZPE): {{{previousTrainingParameters.momentumParams}}}
  - Strength Params (ZPE): {{{previousTrainingParameters.strengthParams}}}
  - Noise Params (ZPE): {{{previousTrainingParameters.noiseParams}}}
  - Coupling Params (ZPE): {{{previousTrainingParameters.couplingParams}}}
  - Quantum Mode: {{{previousTrainingParameters.quantumMode}}}
  - Quantum Circuit Size: {{{previousTrainingParameters.quantumCircuitSize}}} (if quantumMode was true)
  - Label Smoothing: {{{previousTrainingParameters.labelSmoothing}}}
  - Base Config ID (if resumed): {{{previousTrainingParameters.baseConfigId}}}

User\'s Objective for the Next HNN Step:
"{{{hnnObjective}}}"

Your Task:
1. Analyze: Briefly interpret the previousZpeEffects. Are they high, low, varied? How might they relate to the previousTrainingParameters and the hnnObjective?
2. Suggest Parameters: Based on your analysis of the previous ZPE effects (especially noting high/low values for each layer), the previous training parameters (including how previous \`couplingParams\` might have influenced ZPE effects), and the hnnObjective, suggest a *full* set of 'suggestedNextTrainingParameters'. Pay close attention to the interplay between \`couplingParams\` and other ZPE parameters (\`momentumParams\`, \`strengthParams\`, \`noiseParams\`) and how they collectively influence ZPE effects.
   You *must* provide values for *all* parameters listed in the \`suggestedNextTrainingParameters\` object, even if you recommend keeping them the same as the previous job.

Here are the parameters you *must* include in the \`suggestedNextTrainingParameters\` object:
- \`momentumParams\` (array of 6 numbers, 0.0-1.0)
- \`strengthParams\` (array of 6 numbers, 0.0-1.0)
- \`noiseParams\` (array of 6 numbers, 0.0-1.0)
- \`couplingParams\` (array of 6 numbers, 0.0-1.0)
- totalEpochs (integer)
- batchSize (integer)
- learningRate (float, 0.00001-0.1)
- weightDecay (float, 0.0-0.1)
- labelSmoothing (float, 0.0-0.5)
- quantumCircuitSize (integer)
- quantumMode (boolean)
- modelName (string)
- baseConfigId (string or null)

Constraints for ZPE parameters (\`momentumParams\`, \`strengthParams\`, \`noiseParams\`, \`couplingParams\`): values are between 0.0 and 1.0, and each must be an array of 6 values. You *must* include all four of these parameters in your \`suggestedNextTrainingParameters\` object, even if you only suggest changes to some of the values within the arrays.
Learning rate typically between 0.00001 and 0.1.

Output your response in the specified JSON format.
If suggesting changes to array parameters like \`momentumParams\`, \`strengthParams\`, \`noiseParams\`, or \`couplingParams\`, provide the full array of 6 floating-point values (each between 0.0 and 1.0) with the changes.`,
});
const hsQnnParameterAdvisorFlow = ai.defineFlow(
  { // Correct way to define flow name
    name: 'hsQnnParameterAdvisorFlow', // Add the required name property
    inputSchema: HSQNNAdvisorInputSchema,
  },
  async (input: HSQNNAdvisorInput) => {
    const result = await prompt(input);
    if (!result.output) {
      throw new Error('AI failed to generate HNN parameter advice.');
    }

    // Ensure modelName is updated to reflect it\'s a new model
    if (result.output.suggestedNextTrainingParameters && result.output.suggestedNextTrainingParameters.modelName === input.previousTrainingParameters.modelName) {
      result.output.suggestedNextTrainingParameters.modelName = `${input.previousTrainingParameters.modelName}_hnn_next`;
    } else if (result.output.suggestedNextTrainingParameters && !result.output.suggestedNextTrainingParameters.modelName) {
      result.output.suggestedNextTrainingParameters.modelName = `${input.previousTrainingParameters.modelName}_hnn_next`;
    }

    return result.output;
  }
);
