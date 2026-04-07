"""DuckDuckGo web search for persona context enrichment."""

import logging

logger = logging.getLogger(__name__)


def get_persona_context(persona_name: str, max_results: int = 3) -> str:
    """Search the web for background info on a persona.

    Returns concatenated text snippets, or a fallback string if search fails.
    """
    logger.info("Performing web search for: %s", persona_name)
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            query = f"Who is {persona_name}? Background, personality, notable traits."
            results = list(ddgs.text(query, max_results=max_results))

        if results:
            snippets = [f"- {r['body']}" for r in results if r.get("body")]
            if snippets:
                context = "\n".join(snippets)
                logger.info("Found %d snippets for %s", len(snippets), persona_name)
                return context

        logger.warning("No web search results for %s", persona_name)
    except Exception as e:
        logger.error("Error during web search for %s: %s", persona_name, e)

    return "No relevant web context found."
