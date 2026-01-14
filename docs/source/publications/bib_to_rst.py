"""
BibTeX to ReStructuredText (RST) Generator

This script parses BibTeX (.bib) files and generates formatted ReStructuredText (.rst)
files suitable for documentation.

Features:
- Parses multiple configuration sets (All, Technical, Applications).
- Extracts metadata: Title, Authors, Date, Journal, and ADS Links.
- Checks for local images (inside plots) corresponding to the BibCode (e.g., 2020ApJ...123.jpeg)
and embeds them.
- Sorts entries chronologically by Year and Month.

Dependencies:
    pip install bibtexparser
"""

import bibtexparser
import os
import argparse

# All publication configurations, where the different sections are.
all_pubs = {
    'BIB_FILE': 'all_publications/publications.bib',
    'OUTPUT_FILE': 'publications.rst',
    'INTRO_FILE': 'all_publications/intro.inc',  # Optional intro file to prepend
    'HEADER': 'Publications\n============\n\n',  # Fallback header
}

# Configuration for technical publications
technical_pubs = {
    'BIB_FILE': 'technical_publications/publications.bib',
    'OUTPUT_FILE': 'technical_publications.rst',
    'INTRO_FILE': 'technical_publications/intro.inc',
    'HEADER': 'Technical Publications\n======================\n\n',
}

# Configuration for application-specific publications
application_pubs = {
    'BIB_FILE': 'application_publications/publications.bib',
    'OUTPUT_FILE': 'application_publications.rst',
    'INTRO_FILE': 'application_publications/intro.inc',
    'HEADER': 'Application Publications\n========================\n\n',
}

# Directory containing the plot images (filename format: bibcode.jpeg)
IMAGE_DIR = 'plots/'       

def get_author_string(author_field, max_authors=5):
    """
    Parses a BibTeX author string and formats it for display.
    
    Args:
        author_field (str): The raw author string from BibTeX (e.g., "Doe, J. and Smith, A.").
        max_authors (int): Maximum number of authors to display before truncating.

    Returns:
        str: A formatted string. Lists the first 4 authors. 
             If more than 'max_authors', appends "and others".
    """
    if not author_field:
        return "Unknown Authors"
    
    # Split by ' and ', the standard BibTeX author delimiter
    # and define authors list
    authors = author_field.replace('\n', ' ').split(' and ')
    
    # Remove {} from names
    authors = [a.replace('{', '').replace('}', '') for a in authors]
    
    # Clean whitespace from names
    authors = [a.strip() for a in authors]
    
    if len(authors) <= max_authors:
        return ", ".join(authors)
    else:
        return ", ".join(authors[:max_authors]) + " and others"

def get_date_string(entry):
    """
    Formats the publication date string (Month Year).
    
    Args:
        entry (dict): The BibTeX entry dictionary.

    Returns:
        str: Formatted date (e.g., "January 2023") or just Year if month is missing.
    """
    year = entry.get('year', 'n.d.')
    month_raw = entry.get('month', '')
    
    # Map common BibTeX month formats/abbreviations to full names
    month_map = {
        'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April',
        'may': 'May', 'jun': 'June', 'jul': 'July', 'aug': 'August',
        'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December',
        '1': 'January', '2': 'February', '3': 'March', '4': 'April',
        '5': 'May', '6': 'June', '7': 'July', '8': 'August',
        '9': 'September', '10': 'October', '11': 'November', '12': 'December'
    }
    
    # Normalize month key: lowercase, remove braces/spaces
    clean_month = month_raw.lower().strip('{} ')
    
    # Retrieve readable month name, default to original raw string if not found
    month_str = month_map.get(clean_month[:3], month_raw) 
    
    if month_str:
        return f"{month_str} {year}"
    return year

def get_sort_key(entry):
    """
    Generates a sorting key for a BibTeX entry based on Year and Month.
    
    Args:
        entry (dict): The BibTeX entry dictionary.
        
    Returns:
        tuple: (year_int, month_int) for sorting.
    """
    # Parse Year
    try:
        year = int(entry.get('year', '0'))
    except ValueError:
        year = 0

    # Parse Month
    month_raw = entry.get('month', '0').lower().strip('{} ')
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    if month_raw in month_map:
        month = month_map[month_raw]
    elif month_raw.isdigit():
        month = int(month_raw)
    else:
        month = 12 # default if unknown

    return (year, month)

def format_journal_name(journal):
    """
    Replaces LaTeX macro journal names with readable text abbreviations.
    
    Args:
        journal (str): The raw journal string.

    Returns:
        str: The mapped journal name or original if no mapping exists.
    """
    
    journal_map = {
        '\mnras': 'MNRAS',
        '\apj': 'ApJ',
        'The Open Journal of Astrophysics': 'OJA',
        'arXiv e-prints': 'Preprint',
        '\aap': 'A&A',
    }
    
    if journal in journal_map.keys():
        return journal_map[journal]
    else:
        return journal

def get_paper_rst(entry, max_authors=5, is_last=False):
    """
    Generates the ReStructuredText (RST) block for a single publication entry.
    
    Layout:
    - If an image exists: Uses a 'list-table' to display Image (Left) and Metadata (Right).
    - If no image: Standard text block.
    
    Args:
        entry (dict): A single entry from the bibtex database.
        max_authors (int): Maximum number of authors to display.
        is_last (bool): Whether this is the last entry in the list.

    Returns:
        str: The formatted RST string for this entry.
    """
    bibcode = entry.get('ID')
    
    # Prepare Metadata field
    # Remove braces often found in BibTeX titles/journals
    title = entry.get('title', 'Untitled').replace('{', '').replace('}', '')
    
    raw_journal = entry.get('journal', 'Unknown Journal').replace('{', '').replace('}', '')
    journal = format_journal_name(raw_journal)
    
    authors = get_author_string(entry.get('author', ''), max_authors=max_authors)
    date_str = get_date_string(entry)
    
    # Create a link to the NASA ADS Abstract Service
    ads_link = f"https://ui.adsabs.harvard.edu/abs/{bibcode}"
    
    # Extract arXiv ID (standard BibTeX field is 'eprint')
    eprint = entry.get('eprint', '').replace('arXiv:', '') # strip prefix if present
    
    # Build the link line
    links_line = f"`[ADS] <{ads_link}>`__"
    if eprint:
        # Append arXiv link with a separator
        links_line += f" | `[arXiv] <https://arxiv.org/abs/{eprint}>`__"
    
    # Check for Image
    # We expect images to be named exactly as the BibCode (e.g., 2020ApJ...123.jpeg)
    image_filename = f"{bibcode}.jpeg"
    has_image = os.path.exists(os.path.join(IMAGE_DIR, image_filename))
    
    # Build RST for entry    
    rst = ""
    
    if has_image:
        # Side-by-side layout using list-table
        # :class: borderless is useful for Sphinx themes to hide table borders
        rst += f".. list-table::\n"
        rst += f"   :widths: 40 60\n"
        rst += f"   :class: borderless\n\n"
        
        # Column 1: The Image
        rst += f"   * - .. image:: {IMAGE_DIR}/{image_filename}\n"
        rst += f"          :width: 100%\n"  # Fills the 40% column width
        rst += f"          :target: {ads_link}\n" # Makes the image clickable
        
        # Column 2: The Text details
        rst += f"     - **{title}**\n\n"
        rst += f"       Authors: {authors}\n\n"
        rst += f"       {date_str}, *{journal}*\n\n"
        rst += f"       {links_line}\n"
        
    else:
        # Fallback layout: Simple vertical text block
        rst += f"**{title}**\n\n"
        rst += f"Authors: {authors}\n\n"
        rst += f"{date_str}, *{journal}*\n\n"
        rst += f"{links_line}\n"
        
        # Only add the line if it is NOT the last paper
        if not is_last:
            rst += "\n----\n"

    # Add a newline between papers
    rst += "\n"
    
    return rst

def generate_rst(config, max_authors=5):
    """
    Reads a BibTeX file and writes the RST output.
    
    Args:
        config (dict): Dictionary containing file paths (BIB_FILE, OUTPUT_FILE, etc.)
    """
    BIB_FILE = config['BIB_FILE']
    OUTPUT_FILE = config['OUTPUT_FILE']
    INTRO_FILE = config['INTRO_FILE']
    HEADER = config['HEADER']

    print(f"Reading {BIB_FILE}...")
    try:
        with open(BIB_FILE, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)
    except FileNotFoundError:
        print(f"Error: Could not find {BIB_FILE}.")
        return

    entries = bib_database.entries
    # Sort entries by Year (descending) and then Month (descending)
    # We use `get_sort_key` to convert months to integers
    entries.sort(key=get_sort_key, reverse=True)

    rst_content = ""
    
    # If a specific intro file exists (e.g. intro.rst), read it.
    # Otherwise, use the default string defined in HEADER.
    if os.path.exists(INTRO_FILE):
        with open(INTRO_FILE, 'r', encoding='utf-8') as f:
            rst_content += f.read() + "\n\n"
    else:
        rst_content += HEADER

    # Generate RST for each entry
    for ii, entry in enumerate(entries):
        # Check if this is the last item in the list
        is_last = (ii == len(entries) - 1)

        rst_content += get_paper_rst(entry, max_authors=max_authors, is_last=is_last)

    # Write final output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(rst_content)
    
    print(f"Success! Generated {OUTPUT_FILE} with {len(entries)} papers.")

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

if __name__ == '__main__':
    main()