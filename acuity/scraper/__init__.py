"""
ACUITY Framework — Scraper Module

Data collection from Facebook community groups using undetected_chromedriver.

Requires the ``scraper`` extra: ``pip install acuity-framework[scraper]``

Quick Start:
    >>> from acuity.scraper import FacebookScraper
    >>> scraper = FacebookScraper()
    >>> posts = scraper.run(["https://facebook.com/groups/example"])
"""

from .scraper import FacebookScraper

__all__ = ["FacebookScraper"]
