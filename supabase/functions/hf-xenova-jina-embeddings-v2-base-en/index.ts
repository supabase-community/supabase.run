import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.1";

// Create a feature extraction pipeline
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/jina-embeddings-v2-base-en",
  {
    device: "auto",
    quantized: true, // TODO: make it configurable
  },
);

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
