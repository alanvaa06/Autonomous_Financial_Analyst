"""On-demand tools for the Sentiment agent."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

from agents import FAST_MODEL, ToolDef, safe_parse_json


PR_SITES = "site:prnewswire.com OR site:businesswire.com OR site:globenewswire.com"


def _build_haiku(api_key: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=FAST_MODEL, api_key=api_key, temperature=0.1, max_tokens=400,
    )


def _tav_search(tav_client, query: str, days: int, max_results: int = 8) -> dict:
    try:
        result = tav_client.search(
            query=query, max_results=max_results, search_depth="basic", days=days,
        )
    except Exception as exc:
        return {"error": str(exc)[:200]}
    return {"results": [
        {"title": (a.get("title") or "").strip(),
         "url": a.get("url"),
         "snippet": (a.get("content") or "")[:280]}
        for a in (result.get("results", []) or [])
    ]}


def build_sentiment_tools(*, tavily_key: str, api_key: str) -> list[ToolDef]:
    # Instantiate TavilyClient once at build time so the module-level name is
    # resolved inside the patch context when tests call build_sentiment_tools.
    _tav_client = TavilyClient(api_key=tavily_key) if tavily_key else None

    def press_releases_handler(args: dict) -> dict:
        if _tav_client is None:
            return {"error": "tavily_key not configured"}
        ticker = args.get("ticker", "")
        days = int(args.get("days", 14) or 14)
        query = f"{ticker} ({PR_SITES})"
        return _tav_search(_tav_client, query, days)

    def analyst_actions_handler(args: dict) -> dict:
        if _tav_client is None:
            return {"error": "tavily_key not configured"}
        ticker = args.get("ticker", "")
        query = f"{ticker} analyst upgrade downgrade price target"
        return _tav_search(_tav_client, query, days=14)

    def categorize_handler(args: dict) -> dict:
        drivers = args.get("drivers") or []
        if not drivers:
            return {}
        haiku = _build_haiku(api_key)
        prompt = (
            "Classify each driver phrase into exactly one of these categories: "
            "earnings, m&a, regulatory, product, insider, competitor, macro. "
            "Return JSON ONLY: a flat object with each category as a key and "
            "the count of drivers that fall into it as the integer value. "
            "Categories with zero matches MUST still appear with value 0.\n\n"
            "Drivers:\n" + "\n".join(f"- {d}" for d in drivers)
        )
        try:
            resp = haiku.invoke(prompt)
            return safe_parse_json(resp.content)
        except Exception as exc:
            return {"error": str(exc)[:200]}

    return [
        ToolDef(
            name="fetch_press_releases",
            description=(
                "Search issuer press releases (PRNewswire / BusinessWire / "
                "GlobeNewswire) for the ticker over the past `days`. Higher-"
                "signal than aggregator news because the source is the issuer."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days": {"type": "integer", "default": 14},
                },
                "required": ["ticker"],
            },
            handler=press_releases_handler,
        ),
        ToolDef(
            name="fetch_analyst_actions",
            description=(
                "Search for sell-side analyst rating changes and price-target "
                "moves on the ticker over the past 14 days. Use to gauge "
                "consensus drift."
            ),
            input_schema={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
            handler=analyst_actions_handler,
        ),
        ToolDef(
            name="categorize_drivers",
            description=(
                "Categorize a list of driver phrases into a fixed taxonomy: "
                "earnings, m&a, regulatory, product, insider, competitor, "
                "macro. Returns a count per category."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "drivers": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["drivers"],
            },
            handler=categorize_handler,
        ),
    ]
