import "jsr:@supabase/functions-js/edge-runtime.d.ts";

interface WebhookPayload {
  text: string;
}

const model = new Supabase.ai.Session("gte-small");

Deno.serve(async (req) => {
  try {
    // Parse request body
    const payload: WebhookPayload = await req.json();
    const { text } = payload;

    if (!text) {
      throw new Error("Missing required parameter: text");
    }

    // Generate embedding
    const embedding = await model.run(text, {
      mean_pool: true,
      normalize: true,
    });

    // Return embedding
    return new Response(JSON.stringify({ embedding }), {
      status: 200,
    });

    // Handle error
  } catch (error) {
    return new Response(error.message, {
      status: 500,
    });
  }
});
