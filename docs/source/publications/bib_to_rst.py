"""Process bib file for publications page in docs."""

import argparse

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode


def format_entry(entry):
    """Format a single BibTeX entry into readable citation."""
    authors = entry.get("author", "Unknown author")
    year = entry.get("year", "n.d.")
    title = entry.get("title", "Untitled")
    journal = entry.get("journal", "")
    booktitle = entry.get("booktitle", "")
    publisher = entry.get("publisher", "")
    volume = entry.get("volume", "")
    number = entry.get("number", "")
    pages = entry.get("pages", "")

    # Basic APA-like formatting
    formatted = f"{authors} ({year}). *{title}*."

    if journal:
        formatted += f" {journal}"
    elif booktitle:
        formatted += f" In {booktitle}"

    if volume:
        formatted += f", **{volume}**"
        if number:
            formatted += f"({number})"
    if pages:
        formatted += f", {pages}"
    if publisher:
        formatted += f". {publisher}"

    formatted += "."
    return formatted


def convert_bib_to_rst(bib_file, output_rst):
    """Convert a bib file to formatted rst."""
    with open(bib_file, encoding="utf-8") as bibtex_file:
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.load(bibtex_file, parser=parser)

    entries = bib_database.entries
    entries.sort(key=lambda x: x.get("author", ""))

    rst_lines = ["References", "=" * 10, ""]

    for i, entry in enumerate(entries, 1):
        formatted = format_entry(entry)
        rst_lines.append(f"{i}. {formatted}")

    with open(output_rst, "w", encoding="utf-8") as out:
        out.write("\n".join(rst_lines))

    print(f"Converted {len(entries)} references to {output_rst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .bib file to reStructuredText reference list."
    )
    parser.add_argument("bibfile", help="Path to the .bib file")
    parser.add_argument("output", help="Path to output .rst file")

    args = parser.parse_args()
    convert_bib_to_rst(args.bibfile, args.output)
