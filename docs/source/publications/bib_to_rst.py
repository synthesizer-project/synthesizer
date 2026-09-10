"""BibTeX to ReStructuredText (RST) Generator.

This script parses BibTeX (.bib) files and generates formatted
ReStructuredText (.rst) files suitable for documentation.

Notes:
    - Parses multiple configuration sets (All, Technical, Applications).
    - Extracts metadata: Title, Authors, Date, Journal, and ADS Links.
    - Checks for local images (inside plots) corresponding to the
      BibCode (e.g., 2020ApJ...123.jpeg) and embeds them.
    - Sorts entries chronologically by Year and Month.
    - This is intended to be run locally to generate publication
      lists for the documentation. It is not generated automatically
      during documentation builds.

Usage:
    python bib_to_rst.py --max-authors=5
"""

import argparse
import os

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.customization import convert_to_unicode
except ImportError as e:
    raise ImportError(
        "Please install bibtexparser: pip install bibtexparser"
    ) from e
from synthesizer.exceptions import MissingAttribute

# All publication configurations, where the different sections are.
all_pubs = {
    "BIB_FILE": "all_publications/publications.bib",
    "OUTPUT_FILE": "publications.rst",
    # Optional intro file to prepend
    "INTRO_FILE": "all_publications/intro.inc",
    # Fallback header if no intro file is found
    "HEADER": "Publications\n============\n\n",
}

# Configuration for technical publications
technical_pubs = {
    "BIB_FILE": "technical_publications/publications.bib",
    "OUTPUT_FILE": "technical_publications.rst",
    "INTRO_FILE": "technical_publications/intro.inc",
    "HEADER": "Technical Publications\n======================\n\n",
}

# Configuration for application-specific publications
application_pubs = {
    "BIB_FILE": "application_publications/publications.bib",
    "OUTPUT_FILE": "application_publications.rst",
    "INTRO_FILE": "application_publications/intro.inc",
    "HEADER": "Application Publications\n========================\n\n",
}

# At the top of the file, after imports:
script_dir = os.path.dirname(os.path.abspath(__file__))
# Directory containing the plot images (filename format: bibcode.jpeg)
image_dir = "plots"


def get_author_string(author_field: str, max_authors: int = 5) -> str:
    """Parses a BibTeX author string and formats it for display.

    Args:
        author_field (str): The raw author string from BibTeX.
        max_authors (int): Maximum number of authors to display
        before truncating.

    Returns:
        str: A formatted string. Lists the first 4 authors.
             If more than 'max_authors', appends "and others".
    """
    if not author_field:
        return "Unknown Authors"

    # Split by ' and ', the standard BibTeX author delimiter
    # and define authors list
    authors = author_field.replace("\n", " ").split(" and ")

    # Remove {} from names
    authors = [a.replace("{", "").replace("}", "") for a in authors]

    # Clean whitespace from names
    authors = [a.strip() for a in authors]

    if len(authors) <= max_authors:
        return ", ".join(authors)
    else:
        return ", ".join(authors[:max_authors]) + " and others"


def get_date_string(entry: dict) -> str:
    """Formats the publication date string (Month Year).

    Args:
        entry (dict): The BibTeX entry dictionary.

    Returns:
        str: Formatted date (e.g., "January 2023") or just Year
        if month is missing.
    """
    year = entry.get("year", "n.d.")
    month_raw = entry.get("month", "")

    # Map common BibTeX month formats/abbreviations to full names
    month_map = {
        "jan": "January",
        "feb": "February",
        "mar": "March",
        "apr": "April",
        "may": "May",
        "jun": "June",
        "jul": "July",
        "aug": "August",
        "sep": "September",
        "oct": "October",
        "nov": "November",
        "dec": "December",
        "1": "January",
        "2": "February",
        "3": "March",
        "4": "April",
        "5": "May",
        "6": "June",
        "7": "July",
        "8": "August",
        "9": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }

    # Normalize month key: lowercase, remove braces/spaces
    clean_month = month_raw.lower().strip("{} ")

    # Retrieve readable month name
    # Try 3-letter prefix for abbreviations (jan, feb, etc.)
    # or full string for numeric months (1-12)
    month_str = month_map.get(clean_month[:3]) or month_map.get(
        clean_month, month_raw
    )

    if month_str:
        return f"{month_str} {year}"
    return year


def get_sort_key(entry: dict) -> tuple:
    """Generates a sorting key for a BibTeX entry based on Year and Month.

    Args:
        entry (dict): The BibTeX entry dictionary.

    Returns:
        tuple: (year_int, month_int) for sorting.
    """
    # Parse Year
    try:
        year = int(entry.get("year", "0"))
    except ValueError:
        year = 0

    # Parse Month
    month_raw = entry.get("month", "0").lower().strip("{} ")
    month_map_to_int = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    month_key = month_raw[:3] if len(month_raw) >= 3 else month_raw
    if month_key in month_map_to_int:
        month = month_map_to_int[month_key]
    elif month_raw.isdigit():
        month = int(month_raw)
    else:
        month = 1  # default if unknown

    return (year, month)


def format_journal_name(journal: str) -> str:
    """Replaces LaTeX macro journal names with readable text abbreviations.

    Args:
        journal (str): The raw journal string.

    Returns:
        str: The mapped journal name or original if no mapping exists.
    """
    journal_map = {
        r"\mnras": "MNRAS",
        r"\apj": "ApJ",
        "The Open Journal of Astrophysics": "OJA",
        "arXiv e-prints": "Preprint",
        r"\aap": "A&A",
    }

    if journal in journal_map:
        return journal_map[journal]
    else:
        return journal


def get_paper_rst(
    entry: dict,
    max_authors: int = 5,
    number: int = 0,
) -> str:
    """Generates the ReStructuredText (RST) block for a bib entry.

    Layout:
    - Metadata is displayed as a compact line block (numbered title,
      authors, then date, journal and links).
    - If an image exists it is placed inside a collapsible dropdown,
      hidden by default.

    Args:
        entry (dict): A single entry from the bibtex database.
        max_authors (int): Maximum number of authors to display.
        number (int): The index shown before the title. Entries are
            numbered from the bottom of the list upwards, so the oldest
            paper is 1.

    Returns:
        str: The formatted RST string for this entry.
    """
    bibcode = entry.get("ID", "unknown")
    if bibcode == "unknown":
        raise MissingAttribute(
            "ID or bibcode is required in each BibTeX entry."
        )

    # Prepare Metadata field
    # Remove braces often found in BibTeX titles/journals
    title = entry.get("title", "Untitled").replace("{", "").replace("}", "")

    raw_journal = (
        entry.get("journal", "Unknown Journal")
        .replace("{", "")
        .replace("}", "")
    )
    journal = format_journal_name(raw_journal)

    authors = get_author_string(
        entry.get("author", ""), max_authors=max_authors
    )
    date_str = get_date_string(entry)

    # Create a link to the NASA ADS Abstract Service
    ads_link = f"https://ui.adsabs.harvard.edu/abs/{bibcode}"

    # Extract arXiv ID (standard BibTeX field is 'eprint')
    eprint = entry.get("eprint", "").replace(
        "arXiv:", ""
    )  # strip prefix if present

    # Build the link line
    links_line = f"`[ADS] <{ads_link}>`__"
    if eprint:
        # Append arXiv link with a separator
        links_line += f" | `[arXiv] <https://arxiv.org/abs/{eprint}>`__"

    # Check for Image
    # We expect images to be named exactly as the
    # BibCode (e.g., 2020ApJ...123.jpeg)
    image_filename = f"{bibcode}.jpeg"
    image_path = os.path.join(script_dir, image_dir, image_filename)
    has_image = os.path.exists(image_path)
    # Need the relative path from the RST file to the images
    relative_image_path = os.path.join(image_dir, image_filename)

    # Build RST for entry
    # A line block keeps the details on separate lines without the
    # paragraph spacing a sequence of plain paragraphs would introduce.
    # The final line lives in its own container alongside the collapsible
    # figure so that publications.css can render the toggle inline with it.
    rst = ""

    rst += ".. container:: pub-entry\n\n"
    rst += f"    | **{number}. {title}**\n"
    rst += f"    | {authors}\n\n"
    rst += "    .. container:: pub-meta\n\n"
    rst += f"        {date_str}, *{journal}* — {links_line}\n"

    if has_image:
        # Hide the figure behind a collapsible toggle (closed by default)
        rst += "\n        .. collapse:: Figure\n\n"
        rst += f"            .. image:: {relative_image_path}\n"
        rst += "                :width: 60%\n"
        rst += "                :align: center\n"
        # Makes the image clickable
        rst += f"                :target: {ads_link}\n"

    # Add a newline between papers
    rst += "\n"

    return rst


def generate_rst(config: dict, max_authors: int = 5) -> None:
    """Reads a BibTeX file and writes the RST output.

    Args:
        config (dict): Dictionary containing file paths
        (BIB_FILE, OUTPUT_FILE, etc.)
        max_authors (int): Maximum number of authors to display.
    """
    bib_file = os.path.join(script_dir, config["BIB_FILE"])
    output_file = os.path.join(script_dir, config["OUTPUT_FILE"])
    intro_file = os.path.join(script_dir, config["INTRO_FILE"])
    header = config["HEADER"]

    print(f"Reading {bib_file}...")
    try:
        with open(bib_file, "r", encoding="utf-8") as bibtex_file:
            parser = BibTexParser(common_strings=True)
            parser.customization = convert_to_unicode
            bib_database = bibtexparser.load(bibtex_file, parser=parser)
    except FileNotFoundError:
        print(f"Error: Could not find {bib_file}.")
        return
    except Exception as e:
        print(f"Error: Failed to parse {bib_file}: {e}")
        return

    entries = bib_database.entries
    # Sort entries by Year (descending) and then Month (descending)
    # We use `get_sort_key` to convert months to integers
    entries.sort(key=get_sort_key, reverse=True)

    rst_content = ""

    # If a specific intro file exists (e.g. intro.rst), read it.
    # Otherwise, use the default string defined in HEADER.
    if os.path.exists(intro_file):
        with open(intro_file, "r", encoding="utf-8") as f:
            rst_content += f.read() + "\n\n"
    else:
        rst_content += header
    # Generate RST for each entry, numbering from the bottom of the list
    # upwards so the oldest paper is 1. Adding a newer paper then leaves
    # the existing numbers untouched.
    for ii, entry in enumerate(entries):
        rst_content += get_paper_rst(
            entry, max_authors=max_authors, number=len(entries) - ii
        )

    # Write final output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rst_content)

    print(f"Success! Generated {output_file} with {len(entries)} papers.")


def main() -> None:
    """Command-line interface for converting .bib to .rst.

    Parses arguments, then converts the provided BibTeX file to a
    reStructuredText publication list using release-date (descending) order.

    Usage:
        python bib_to_rst.py --max-authors=5

    Notes:
        - The script processes three publication configurations:
          all publications, technical publications, and application
          publications.
        - Each configuration reads from its own .bib file and writes
          to its own .rst output file.

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
    parser.add_argument(
        "--max-authors",
        type=int,
        default=5,
        help="Maximum authors to list before 'and others' (default: 5)",
    )
    args = parser.parse_args()

    generate_rst(all_pubs, max_authors=args.max_authors)
    generate_rst(technical_pubs, max_authors=args.max_authors)
    generate_rst(application_pubs, max_authors=args.max_authors)


if __name__ == "__main__":
    main()
