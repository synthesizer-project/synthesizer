"""Process a .bib file into a .rst publication list.

This script preserves APA-like formatting, augments entries with inline
hyperlinks (DOI, arXiv, URL/ADS/PDF), cleans author names and titles, and
sorts by release date using `date` (preferred) or `year` + `month` (+`day`)
when available.

Example:
    Convert a BibTeX file to an .rst list:

        python bib_to_rst.py publications.bib publications.rst

Notes:
    - Sorting (release order): entries are sorted by (year, month, day)
      descending. If only `year` is given, the month defaults to December (12)
      and day to 31 so that less specific dates still appear late in the year.
    - Links: constructed from `doi`, `eprint` (+ `archiveprefix`/`eprinttype`
      for arXiv), `pdf`, and `adsurl`/`url`/first URL in `howpublished`.
    - Formatting: titles are italicized, journals/booktitles italicized, and
      volume is bold. Your original spacing for volume/number—`**vol** (no)`—
      is preserved.
"""

import argparse
import re
from typing import List, Tuple

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.customization import convert_to_unicode
except ImportError as e:
    raise ImportError(
        "Please install bibtexparser: pip install bibtexparser"
    ) from e

# Constants and Regular Expressions
BRACE_RE = re.compile(r"[{}]")
WS_RE = re.compile(r"\s+")
AND_SPLIT_RE = re.compile(r"\s+and\s+", re.IGNORECASE)
MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def rst_escape(text: str) -> str:
    r"""Escape characters that could break inline markup in reStructuredText.

    Backslashes, backticks, asterisks, and underscores are escaped. Excess or
    odd whitespace is collapsed to a single space.

    Args:
        text: The input string to escape.

    Returns:
        A version of `text` safe for inline reST markup.
    """
    if not text:
        return ""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`").replace("*", "\\*").replace("_", "\\_")
    return WS_RE.sub(" ", text).strip()


def clean_title(title: str) -> str:
    """Remove BibTeX capitalization braces from a title.

    Args:
        title: A title string that may include `{}` braces from BibTeX.

    Returns:
        The title with `{}` removed.
    """
    return BRACE_RE.sub("", title or "").strip()


def name_to_last_fm(name: str) -> str:
    """Convert a BibTeX-ish name to the form ``'Last, F. M.'``.

    Handles both ``'Last, First Middle'`` and ``'First Middle Last'`` forms.

    Args:
        name: The raw name string from BibTeX.

    Returns:
        A string formatted as ``'Last, F. M.'``. When the input is empty,
        returns ``'Unknown'``.
    """
    name = name.strip()
    if not name:
        return "Unknown"
    if "," in name:
        last, rest = [p.strip() for p in name.split(",", 1)]
        initials = " ".join(f"{p[0]}." for p in rest.split() if p)
        return f"{last}, {initials}".strip().rstrip(",")
    parts = name.split()
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
    return f"{last}, {initials}".strip().rstrip(",")


def format_authors(
    author_field: str, max_authors: int = 10, et_al: str = "et al."
) -> str:
    """Format a BibTeX ``author`` field into a compact authors list.

    Converts each name to ``'Last, F. M.'``. If the number of authors exceeds
    `max_authors`, truncates and appends ``'et al.'``.

    Args:
        author_field: The raw BibTeX ``author`` field (names separated by
            ``and``).
        max_authors: Maximum number of authors to list before truncation.
        et_al: The suffix to indicate truncation (e.g., ``'et al.'``).

    Returns:
        A formatted authors string.
    """
    if not author_field:
        return "Unknown author"
    authors_raw = [a for a in AND_SPLIT_RE.split(author_field) if a.strip()]
    authors = [name_to_last_fm(a) for a in authors_raw]
    if len(authors) <= max_authors:
        return ", ".join(authors[:-1]) + (
            f" & {authors[-1]}" if len(authors) > 1 else authors[0]
        )
    shown = authors[:max_authors]
    return ", ".join(shown) + f", {et_al}"


def parse_month(m: str) -> int:
    """Convert a month token to an integer in ``[1, 12]``.

    Accepts numeric strings (``"3"``, ``"12"``) and common BibTeX tokens
    (``"jan"``, ``"february"``). Unknown or out-of-range values default to 12.

    Args:
        m: Month token from BibTeX.

    Returns:
        Integer month in the range 1–12, defaulting to 12 when unknown.
    """
    if not m:
        return 12
    m = m.strip().lower()
    if m.isdigit():
        mi = int(m)
        return mi if 1 <= mi <= 12 else 12
    m = m.strip("{}")
    return MONTH_MAP.get(m, 12)


def parse_date_tuple(entry: dict) -> Tuple[int, int, int]:
    """Extract a sortable (year, month, day) tuple for a BibTeX entry.

    Preference order:
      1. ``date`` field in ``YYYY[-MM[-DD]]`` format (or ``released``)
      2. ``year`` (string with digits) plus ``month`` (+ optional ``day``)

    Missing components default to a *late* date (Dec 31) so less-specific
    entries still appear late in the year when sorting by recency.

    Args:
        entry: A parsed BibTeX entry dict.

    Returns:
        A tuple ``(year, month, day)`` where all components are integers.
        If the year is missing/invalid, returns ``(0, 1, 1)`` so that such
        entries sort to the very bottom.
    """
    y = 0
    m = 12
    d = 31

    ds = entry.get("date", "") or entry.get("released", "")
    if ds:
        mobj = re.match(r"^\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?\s*$", ds)
        if mobj:
            y = int(mobj.group(1))
            if mobj.group(2):
                m = max(1, min(12, int(mobj.group(2))))
            if mobj.group(3):
                d = max(1, min(31, int(mobj.group(3))))

    if y == 0:
        y_field = re.sub(r"[^\d]", "", entry.get("year", ""))
        y = int(y_field) if y_field else 0
        m = parse_month(entry.get("month", "")) if y else 12
        try:
            d = int(entry.get("day", "31"))
            d = max(1, min(31, d))
        except Exception:
            d = 31

    return (y, m, d) if y > 0 else (0, 1, 1)


def pick_url(entry: dict) -> str:
    """Select a generic URL for an entry when DOI/arXiv are unavailable.

    Preference order: ``adsurl`` > ``url`` > first URL-like token in
    ``howpublished``.

    Args:
        entry: A parsed BibTeX entry dict.

    Returns:
        A URL string if available, otherwise an empty string.
    """
    for key in ("adsurl", "url"):
        u = entry.get(key, "")
        if u and u.lower().startswith(("http://", "https://")):
            return u
    hp = entry.get("howpublished", "")
    if "http://" in hp or "https://" in hp:
        m = re.search(r"(https?://\S+)", hp)
        if m:
            return m.group(1).rstrip(".,);")
    return ""


def format_links(entry: dict) -> str:
    """Construct a compact set of inline links for a BibTeX entry.

    Builds inline reST links to DOI, arXiv, PDF, and a generic URL (ADS/URL)
    when available. Links are separated by ``·`` and prefixed with two spaces
    to visually offset them from the citation text.

    Args:
        entry: A parsed BibTeX entry dict.

    Returns:
        A string beginning with two spaces and containing inline links, or an
        empty string when no links are available.
    """
    parts: List[str] = []

    doi = entry.get("doi", "").strip()
    if doi:
        doi = doi.replace("https://doi.org/", "").replace(
            "http://doi.org/", ""
        )
        parts.append(f"`doi:{rst_escape(doi)} <https://doi.org/{doi}>`_")

    eprint = entry.get("eprint", "").strip()
    ap = entry.get("archiveprefix", entry.get("archivePrefix", "")).lower()
    etype = entry.get("eprinttype", "").lower()
    if eprint and (
        ap == "arxiv"
        or etype == "arxiv"
        or re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", eprint)
    ):
        parts.append(
            f"`arXiv:{rst_escape(eprint)} <https://arxiv.org/abs/{eprint}>`_"
        )

    pdf = entry.get("pdf", "").strip()
    if pdf.startswith(("http://", "https://")):
        parts.append(f"`PDF <{rst_escape(pdf)}>`_")

    url = (entry.get("adsurl") or "").strip()
    if not url:
        u = (entry.get("url") or "").strip()
        if u.startswith(("http://", "https://")):
            url = u
    if not url:
        hp = entry.get("howpublished", "")
        if "http://" in hp or "https://" in hp:
            m = re.search(r"(https?://\S+)", hp)
            if m:
                url = m.group(1).rstrip(".,);")

    if url:
        # Label as ADS if it came from adsurl or looks like ADS; else 'link'
        is_ads = bool(entry.get("adsurl")) or (
            "adsabs.harvard.edu" in url.lower()
        )
        label = "ADS" if is_ads else "link"
        parts.append(f"`{label} <{rst_escape(url)}>`_")

    return "  " + " · ".join(parts) if parts else ""


def format_journal(entry: dict) -> str:
    """Build the journal/booktitle, volume(issue), pages, publisher string.

    Journals and booktitles are italicized, volumes are bold, and issue numbers
    are parenthesized with your preferred spacing style.

    Args:
        entry: A parsed BibTeX entry dict.

    Returns:
        A venue string (without a trailing period). Returns an empty string if
        neither ``journal`` nor ``booktitle`` nor other venue details are
        present.
    """
    journal = rst_escape(entry.get("journal", ""))
    booktitle = rst_escape(entry.get("booktitle", ""))
    volume = rst_escape(entry.get("volume", ""))
    number = rst_escape(entry.get("number", ""))
    pages = rst_escape(entry.get("pages", ""))
    publisher = rst_escape(entry.get("publisher", ""))

    bits: List[str] = []
    if journal:
        bits.append(f"*{journal}*")
    elif booktitle:
        bits.append(f"In *{booktitle}*")

    if volume:
        vol = f"**{volume}**"
        if number:
            vol += f" ({number})"  # retain your space + parentheses style
        bits.append(vol)

    if pages:
        bits.append(pages)

    if publisher and not journal:
        bits.append(publisher)

    return ", ".join(bits)


def format_entry(entry: dict, max_authors: int = 10) -> str:
    """Format a single BibTeX entry into a readable, link-rich citation.

    The output follows your APA-like layout:

        ``Authors (Year). *Title*. Venue.  Links``

    Where *Links* (if present) are DOI/arXiv/PDF/URL and are separated by
    ``·``.

    Args:
        entry: The parsed BibTeX entry.
        max_authors: Maximum number of authors to list before truncation to
            ``'et al.'``.

    Returns:
        A single-line formatted citation string suitable for numbered lists.
    """
    authors = format_authors(entry.get("author", ""), max_authors=max_authors)
    year = entry.get("year", "n.d.").strip()
    title = rst_escape(clean_title(entry.get("title", "Untitled")))
    venue = format_journal(entry)

    formatted = f"{authors} ({year}). *{title}*."
    if venue:
        formatted += f" {venue}."
    formatted += format_links(entry)
    return formatted


def convert_bib_to_rst(
    bib_file: str, output_rst: str, max_authors: int = 10
) -> None:
    """Convert a BibTeX file to a release-ordered .rst publication list.

    Reads entries from ``bib_file``, sorts them by descending release date
    (year, month, day), formats each entry with venue and hyperlinks, and
    writes a numbered list to ``output_rst`` with a top-level "Publications"
    header.

    Args:
        bib_file: Path to the input ``.bib`` file.
        output_rst: Path to the output ``.rst`` file.
        max_authors: Maximum number of authors to list before truncation.

    Returns:
        None. Writes an ``.rst`` file to ``output_rst``.

    Raises:
        FileNotFoundError: If ``bib_file`` does not exist.
        ImportError: If ``bibtexparser`` is not installed.
        UnicodeDecodeError: If files cannot be decoded as UTF-8.
    """
    with open(bib_file, encoding="utf-8") as bibtex_file:
        parser = BibTexParser(common_strings=True)
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.load(bibtex_file, parser=parser)

    entries = bib_database.entries

    # Precompute sort keys: release order (most recent first)
    for e in entries:
        y, m, d = parse_date_tuple(e)
        e["_sort_date"] = (y, m, d)

        # Stable tie-breakers: first author's last name, then title
        first_author = (
            AND_SPLIT_RE.split(e.get("author", "").strip())[0]
            if e.get("author")
            else ""
        )
        if "," in first_author:
            e["_first_last"] = first_author.split(",", 1)[0].strip().lower()
        else:
            e["_first_last"] = (
                first_author.split()[-1] if first_author else ""
            ).lower()
        e["_title_sort"] = clean_title(e.get("title", "")).lower()

    # Sort: date desc, then first author asc, then title asc
    entries.sort(
        key=lambda x: (
            -x["_sort_date"][0],
            -x["_sort_date"][1],
            -x["_sort_date"][2],
            x["_first_last"],
            x["_title_sort"],
        )
    )

    rst_lines = ["Publications", "=" * len("Publications"), ""]

    for i, entry in enumerate(entries, 1):
        formatted = format_entry(entry, max_authors=max_authors)
        rst_lines.append(f"{i}. {formatted}")

    with open(output_rst, "w", encoding="utf-8") as out:
        # Ensure trailing newline for POSIX friendliness
        out.write("\n".join(rst_lines) + "\n")

    print(f"Converted {len(entries)} references to {output_rst}")


def main() -> None:
    """Command-line interface for converting .bib to .rst.

    Parses arguments, then converts the provided BibTeX file to a
    reStructuredText publication list using release-date (descending) order.

    Usage:
        python bib_to_rst.py publications.bib publications.rst

    Args:
        None. Arguments are read from ``sys.argv``.

    Returns:
        None. Writes to the output path given on the command line.

    Raises:
        SystemExit: If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser(
        description="Convert .bib file to a release-ordered, hyperlink-rich "
        "reStructuredText reference list."
    )
    parser.add_argument("bibfile", help="Path to the .bib file")
    parser.add_argument("output", help="Path to output .rst file")
    parser.add_argument(
        "--max-authors",
        type=int,
        default=10,
        help="Maximum authors to list before 'et al.' (default: 10)",
    )
    args = parser.parse_args()
    convert_bib_to_rst(args.bibfile, args.output, max_authors=args.max_authors)


if __name__ == "__main__":
    main()
