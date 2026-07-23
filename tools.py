from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
import os
import wikipedia

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")


if not api_key:
    api_key = st.secrets.get("TAVILY_API_KEY")
web_tools = TavilySearch(max_results=3)


# arxiv_wrapper = ArxivAPIWrapper(top_k_results=3,doc_content_chars_max=500)
# arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

import arxiv
from langchain_core.tools import tool


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for scientific and academic research papers."""

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []

    for paper in client.results(search):
        authors = ", ".join(
            author.name for author in paper.authors
        )

        results.append(
            f"""
Title: {paper.title}

Authors: {authors}

Published: {paper.published}

Summary:
{paper.summary}

Paper URL:
{paper.entry_id}
"""
        )

    if not results:
        return "No arXiv papers found."

    return "\n\n".join(results)



# Set a custom User-Agent
wikipedia.wikipedia.USER_AGENT = (
    "MyLangChainApp/1.0 (dhananjaykarena@gmail.com)"
)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

Tools = [web_tools,search_arxiv,wiki]


