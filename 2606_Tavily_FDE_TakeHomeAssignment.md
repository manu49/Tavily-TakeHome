# Tavily - FDE - Take Home Assignment

## Option 1: Improve an existing application

We've built a small Python CLI (`starter_agent.py`) that demonstrates how Tavily can be used in a simple agentic workflow. It uses LangChain's `create_agent`, a Nebius-hosted chat model, and the Tavily search tool. The system is intentionally basic, has limitations, and is not optimized.

To run it, create a Tavily account and API key at <https://app.tavily.com>. Then create a Nebius API key at <https://tokenfactory.nebius.com> at Token Factory; Alternatively, you are free to use any inference provider and model of your choice.

Set both keys locally, either in your shell or in a `.env` file:

```bash
TAVILY_API_KEY="tvly-..."
NEBIUS_API_KEY="..."
```

Then run:

```bash
uv run starter_agent.py "What changed in the AI search market this year?"
```

Your task is to meaningfully improve it in a way that creates clear business and/or technical value for a real user or customer.

### Example directions

- Adapt it to a specific customer workflow
- Add a useful integration
- Improve retrieval quality
- Improve source handling and citations
- Add an evaluation loop
- Introduce a context engineering improvement
- Improve observability/debuggability

### Deliverables

- GitHub repository with your implementation. Do not include the `starter_agent.py` file in the repository.
- Brief technical statement explaining your approach, thought process, and the value it creates from a technical and/or business perspective
- A record of how you built this — coding agent chat history, session logs, or equivalent — in whatever format you are comfortable sharing. Common formats include Markdown files, video recordings, or shareable links using platforms like [Traces.com](https://traces.com).

## Option 2: Create an explainer

Develop an explainer that demonstrates an interesting technical concept related to AI engineering. This is your opportunity to showcase clear documentation, communication skills, and creativity. Diagrams and interactivity is highly encouraged.

The explainer should feel useful to a developer, partner, or technical customer who wants to understand or apply the concept.

### Example topics

- A unique context engineering approach for handling deep research tasks
- An evaluation framework for agentic systems with retrieval
- A practical integration pattern using Tavily with another AI engineering tool or framework
- Any other relevant AI engineering concept

### Deliverables

- Your explainer in whatever format you are comfortable sharing. Assume you are publishing something for public consumption.
- Brief technical statement explaining your approach, thought process, and the value it creates from a technical and/or business perspective
- A record of how you built this — coding agent chat history, session logs, or equivalent — in whatever format you are comfortable sharing. Common formats include Markdown files, video recordings, or shareable links using platforms like [Traces.com](https://traces.com).

## Requirements

- We expect this to take roughly 4–6 hours. We'd rather see a small thing done well than a large thing done loosely.
- Demonstrate deep understanding of AI engineering and the ecosystem.
- Less is more: emphasis on simplicity, clarity, and quality.
- We expect you'll use AI tools — we want to see that you direct them well. Submissions that read as unverified model output will be penalized heavily.
- Bonus points for submissions that use or align with industry-standard approaches for tracing and observability.
