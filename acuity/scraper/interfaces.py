"""
ACUITY Framework — Scraper Interfaces

Abstract base classes defining pluggable extension points for data
collection. Third-party code can implement these interfaces to provide
custom data sources without modifying ACUITY's source.

Example:
    >>> from acuity.scraper.interfaces import DataSource
    >>> class MySource(DataSource):
    ...     def fetch_posts(self, sources, max_posts=500):
    ...         return [{"text": "Sample post", "poster": "User"}]
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract interface for data sources that provide post text.

    Implement this class to provide a custom data source (e.g., CSV file,
    database, API) that can be injected into
    :class:`~acuity.extraction.pipeline.ExtractionPipeline` via its
    ``data_source`` constructor parameter.

    The data source receives a list of source identifiers (URLs, file paths,
    etc.) and returns a list of post dictionaries.

    Example:
        >>> class DatabaseSource(DataSource):
        ...     def fetch_posts(self, sources, max_posts=500):
        ...         # Query database for posts
        ...         return [{"text": "post text", "poster": "author"}]
    """

    @abstractmethod
    def fetch_posts(
        self,
        sources: list[str],
        max_posts: int = 500,
    ) -> list[dict]:
        """Fetch posts from the data source.

        Args:
            sources: List of source identifiers. The meaning depends on the
                implementation (e.g., URLs, file paths, database table names).
            max_posts: Maximum number of posts to return.

        Returns:
            A list of dictionaries, each with at least a ``"text"`` key
            containing the post content. Optional keys include:

            - ``"poster"``: Name of the post author.
            - ``"scraped_at"``: Timestamp or date string of when the post
              was collected.
        """
        ...
