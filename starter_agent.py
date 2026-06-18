# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "langchain>=1.0.0",
#   "langchain-nebius>=0.1.0",
#   "langchain-tavily>=0.2.0",
#   "python-dotenv>=1.0.0",
#   "rich>=13.0.0",
#   "typer>=0.12.0",
# ]
# ///
"""
Minimal Tavily + LangChain agent CLI.

Setup:
  1. Create a Tavily API key: https://app.tavily.com
  2. Create a Nebius API key: https://tokenfactory.nebius.com
     You will receive a code for free credits.
  3. Export both keys or add them to a .env file:
       TAVILY_API_KEY="tvly-..."
       NEBIUS_API_KEY="..."
  4. Run:
       uv run starter_agent.py "What are the latest trends in agent evaluation?"

This is intentionally simple and uses streaming to improve user experience. It creates a LangChain agent with a Nebius-hosted
chat model and the Tavily search tool, then streams the agent run in the console.

You are not required to follow the below sample code. 
We are looking for meaningful improvement and clear explanation of your approach, whatever that may be.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Optional

import typer
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch
from pydantic import PrivateAttr
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


load_dotenv()


class FixedChatNebius(ChatNebius):
    """Works around a Nebius/vLLM streaming bug in gpt-oss models: the final
    argument fragment of a tool call is sometimes reported under a new
    `index` instead of the one it belongs to. LangChain merges tool_call
    chunks by index, so the stray fragment turns into a second, malformed
    tool call (no id/name) that the API rejects on the next turn. Chunks
    with no id/name are re-attached to the most recently started tool call
    instead of trusting the server-reported index.
    """

    _active_tool_call_index: Optional[int] = PrivateAttr(default=None)

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return generation_chunk

        tool_call_chunks = getattr(generation_chunk.message, "tool_call_chunks", None)
        if not tool_call_chunks:
            return generation_chunk

        for tool_call_chunk in tool_call_chunks:
            if tool_call_chunk.get("id"):
                self._active_tool_call_index = tool_call_chunk.get("index")
            elif self._active_tool_call_index is not None:
                tool_call_chunk["index"] = self._active_tool_call_index

        return generation_chunk

app = typer.Typer(add_completion=False)
console = Console()

SYSTEM_PROMPT = """You are a concise research assistant.
Use Tavily search when you need current or factual web information.
Answer the user's question directly and include source URLs when available.
"""


def require_env(name: str, instructions: str) -> None:
    if os.getenv(name):
        return
    console.print(f"[bold red]Missing {name}[/bold red]")
    console.print(instructions)
    raise typer.Exit(code=1)


def message_text(message: Any) -> str:
    """Extract streamed text from a LangChain message or message chunk."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def flush() -> None:
    console.file.flush()


def truncate(value: Any, limit: int = 900) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def print_tool_call(name: str, args: Any) -> None:
    console.print()
    console.print(
        Panel(
            Text(truncate(args, limit=700)),
            title=f"Tool call: {name}",
            border_style="yellow",
        )
    )


def format_tool_result(content: Any) -> str:
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return truncate(content)

    if not isinstance(payload, dict) or "results" not in payload:
        return truncate(payload)

    lines = [f"Query: {payload.get('query', '')}", ""]
    for index, result in enumerate(payload.get("results", [])[:5], start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = " ".join(result.get("content", "").split())
        lines.append(f"{index}. {title}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {truncate(snippet, limit=220)}")
        lines.append("")
    return "\n".join(lines).strip()


def print_tool_result(message: Any) -> None:
    name = getattr(message, "name", None) or "tool"
    content = format_tool_result(getattr(message, "content", ""))
    console.print()
    console.print(
        Panel(
            Text(content),
            title=f"Tool result: {name}",
            border_style="yellow",
        )
    )


@app.command()
def main(
    question: Annotated[list[str], typer.Argument(help="Question")],
    model: Annotated[
        str,
        typer.Option(help="Model name"),
    ] = "openai/gpt-oss-120b",
) -> None:
    """Ask a question and stream a small LangChain agent that searches with Tavily."""

    require_env(
        "TAVILY_API_KEY",
        "Create one at https://app.tavily.com, then run: export TAVILY_API_KEY='tvly-...'",
    )
    require_env(
        "NEBIUS_API_KEY",
        "Create one at https://tokenfactory.nebius.com, then run: export NEBIUS_API_KEY='...'",
    )

    question_text = " ".join(question)

    chat_model = FixedChatNebius(model=model, streaming=True)
    search_tool = TavilySearch()

    agent = create_agent(
        model=chat_model,
        tools=[search_tool],
        system_prompt=SYSTEM_PROMPT,
    )

    console.print(Panel.fit(question_text, title="Question", border_style="cyan"))
    console.rule("[bold blue]Agent stream")

    tool_buffers: dict[str, dict[str, str]] = {}
    printed_tool_calls: set[str] = set()
    assistant_started = False
    last_event_was_text = False

    try:
        stream = agent.stream(
            {"messages": [{"role": "user", "content": question_text}]},
            stream_mode=["messages", "updates"],
        )

        for mode, data in stream:
            if mode == "messages":
                message, _metadata = data

                if getattr(message, "type", None) == "tool":
                    continue

                tool_call_chunks = getattr(message, "tool_call_chunks", []) or []
                if tool_call_chunks:
                    if last_event_was_text:
                        console.print()
                        last_event_was_text = False

                    for chunk in tool_call_chunks:
                        key = str(chunk.get("id") or chunk.get("index") or "tool_call")
                        buffer = tool_buffers.setdefault(key, {"name": "", "args": ""})

                        if chunk.get("name"):
                            buffer["name"] += chunk["name"]
                        if chunk.get("args"):
                            buffer["args"] += chunk["args"]

                        if key not in printed_tool_calls and buffer["name"]:
                            printed_tool_calls.add(key)
                            console.print(
                                f"\n[bold yellow]Tool call[/bold yellow] [yellow]{buffer['name']}[/yellow]",
                                highlight=False,
                            )
                            console.print("[dim yellow]args: [/dim yellow]", end="")

                        if chunk.get("args"):
                            console.print(
                                chunk["args"], style="yellow", end="", highlight=False
                            )
                            flush()
                    continue

                text = message_text(message)
                if text:
                    if not assistant_started:
                        if printed_tool_calls:
                            console.print()
                        console.print("\n[bold green]Assistant[/bold green]")
                        assistant_started = True
                    console.print(text, end="", highlight=False, markup=False)
                    flush()
                    last_event_was_text = True

            elif mode == "updates":
                for node_update in data.values():
                    for message in node_update.get("messages", []):
                        if getattr(message, "type", None) == "ai":
                            for tool_call in getattr(message, "tool_calls", []) or []:
                                key = str(
                                    tool_call.get("id")
                                    or tool_call.get("name")
                                    or "tool_call"
                                )
                                if key not in printed_tool_calls:
                                    printed_tool_calls.add(key)
                                    print_tool_call(
                                        tool_call.get("name", "tool"),
                                        tool_call.get("args", {}),
                                    )

                        if getattr(message, "type", None) == "tool":
                            if last_event_was_text:
                                console.print()
                                last_event_was_text = False
                            print_tool_result(message)

        if last_event_was_text:
            console.print()

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        raise typer.Exit(code=130) from None
    except Exception as exc:
        console.print(f"\n[bold red]Agent run failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
