// Follow this setup guide to integrate the Deno language server with your editor:
import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.1";

// Create a feature extraction pipeline
const extractor = await pipeline(
  "feature-extraction",
  "nomic-ai/nomic-embed-text-v1",
  {
    device: "auto",
    quantized: true, // TODO: make it configurable
  },
);

// Compute sentence embeddings

Deno.serve(async (req) => {
  const { input } = await req.json();

  const texts = [input];
  const embeddings = await extractor(texts, {
    // TODO: make these params configurable
    pooling: "mean",
    normalize: true,
  });

  return new Response(JSON.stringify(embeddings), {
    headers: { "Content-Type": "application/json" },
    status: 200,
  });
});

/* To invoke locally:

  1. Run `supabase start` (see: https://supabase.com/docs/reference/cli/supabase-start)
  2. Make an HTTP request:

  curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/hf-nomic-ai-nomic-embed-text-v1' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0' \
    --header 'Content-Type: application/json' \
    --data '{"name":"Functions"}'

*/
