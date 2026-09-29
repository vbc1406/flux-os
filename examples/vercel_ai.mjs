// Vercel AI SDK (Node) — every provider flavour works against flux-os.
//   npm i ai @ai-sdk/openai-compatible @ai-sdk/openai @ai-sdk/anthropic zod
//   flux-os serve && node examples/vercel_ai.mjs
import { generateText, tool, stepCountIs } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { z } from 'zod';

const flux = createOpenAICompatible({ name: 'flux-os', baseURL: 'http://localhost:8000/v1', apiKey: 'unused' });
// Also works: createOpenAI({ baseURL }) (Responses API) and createAnthropic({ baseURL: 'http://localhost:8000/v1' })

const { text } = await generateText({
  model: flux('auto'),
  prompt: "What's the weather in Paris?",
  tools: {
    get_weather: tool({
      description: 'Get the weather for a city',
      inputSchema: z.object({ city: z.string() }),
      execute: async ({ city }) => `18C and sunny in ${city}`,
    }),
  },
  stopWhen: stepCountIs(3),
});
console.log(text);
