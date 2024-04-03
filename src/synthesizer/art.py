"""A module containing ASCII art for pretty outputs.

Example usage:

    from synthesizer.art import Art
    print(Art.galaxy)
"""

from typing import List


class Art:
    """A class containing art."""

    synthesizer: str = (
        "+-+-+-+-+-+-+-+-+-+-+-+\n"
        "|S|Y|N|T|H|E|S|I|Z|E|R|\n"
        "+-+-+-+-+-+-+-+-+-+-+-+\n"
    )

    galaxy2: str = (
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⡀⠒⠒⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⢀⣤⣶⡾⠿⠿⠿⠿⣿⣿⣶⣦⣄⠙⠷⣤⡀⠀⠀⠀⠀\n"
        "⠀⠀⠀⣠⡾⠛⠉⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⣷⣄⠘⢿⡄⠀⠀⠀\n"
        "⠀⢀⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠐⠂⠠⢄⡀⠈⢿⣿⣧⠈⢿⡄⠀⠀\n"
        "⢀⠏⠀⠀⠀⢀⠄⣀⣴⣾⠿⠛⠛⠛⠷⣦⡙⢦⠀⢻⣿⡆⠘⡇⠀⠀\n"
        "⠀⠀⠀⠀⡐⢁⣴⡿⠋⢀⠠⣠⠤⠒⠲⡜⣧⢸⠄⢸⣿⡇⠀⡇⠀⠀\n"
        "⠀⠀⠀⡼⠀⣾⡿⠁⣠⢃⡞⢁⢔⣆⠔⣰⠏⡼⠀⣸⣿⠃⢸⠃⠀⠀\n"
        "⠀⠀⢰⡇⢸⣿⡇⠀⡇⢸⡇⣇⣀⣠⠔⠫⠊⠀⣰⣿⠏⡠⠃⠀⠀⢀\n"
        "⠀⠀⢸⡇⠸⣿⣷⠀⢳⡈⢿⣦⣀⣀⣀⣠⣴⣾⠟⠁⠀⠀⠀⠀⢀⡎\n"
        "⠀⠀⠘⣷⠀⢻⣿⣧⠀⠙⠢⠌⢉⣛⠛⠋⠉⠀⠀⠀⠀⠀⠀⣠⠎⠀\n"
        "⠀⠀⠀⠹⣧⡀⠻⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠃⠀⠀\n"
        "⠀⠀⠀⠀⠈⠻⣤⡈⠻⢿⣿⣷⣦⣤⣤⣤⣤⣤⣴⡾⠛⠉⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠈⠙⠶⢤⣈⣉⠛⠛⠛⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
    )

    galaxy: str = (
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⡀⠒⠒⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⢀⣤⣶⡾⠿⠿⠿⠿⣿⣿⣶⣦⣄⠙⠷⣤⡀⠀⠀⠀⠀\n"
        "⠀⠀⠀⣠⡾⠛⠉⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⣷⣄⠘⢿⡄⠀⠀⠀\n"
        "⠀⢀⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠐⠂⠠⢄⡀⠈⢿⣿⣧⠈⢿⡄⠀⠀\n"
        "⢀⠏⠀⠀⠀⢀⠄⣀⣴⣾⠿⠛⠛⠛⠷⣦⡙⢦⠀⢻⣿⡆⠘⡇⠀⠀\n"
        "⠀⠀⠀+-+-+-+-+-+-+-+-+-+-+-+⡇⠀⠀\n"
        "⠀⠀⠀|S|Y|N|T|H|E|S|I|Z|E|R|⠃⠀⠀\n"
        "⠀⠀⢰+-+-+-+-+-+-+-+-+-+-+-+⠀⠀⠀\n"
        "⠀⠀⢸⡇⠸⣿⣷⠀⢳⡈⢿⣦⣀⣀⣀⣠⣴⣾⠟⠁⠀⠀⠀⠀⢀⡎\n"
        "⠀⠀⠘⣷⠀⢻⣿⣧⠀⠙⠢⠌⢉⣛⠛⠋⠉⠀⠀⠀⠀⠀⠀⣠⠎⠀\n"
        "⠀⠀⠀⠹⣧⡀⠻⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠃⠀⠀\n"
        "⠀⠀⠀⠀⠈⠻⣤⡈⠻⢿⣿⣷⣦⣤⣤⣤⣤⣤⣴⡾⠛⠉⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠈⠙⠶⢤⣈⣉⠛⠛⠛⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
    )

    blackhole: str = (
        "                                                               \n"
        "                     ´```´``````````````´                      \n"
        "               `´```````````´´`´´¨´´```````````                \n"
        "           ````````````´¨¨···…………··¨¨¨´´``````````             \n"
        "        ```````````´¨··¸¸’‚:›‹¹¹‹;’‚˜¸…·¨¨´``````````          \n"
        "      `````````´´´¨·¸˜‘›º—!¿jo¤îï{•^³°;‚ˆ¸·¨¨´´````````        \n"
        "     ````````´´´¨·¸˜’­„¿‰uusssssöö¢u<«^³‹‘˜¸·¨¨´´´````````      \n"
        "    ````````´´¨··¸’;³(¤‰uò%ízí1It‰¢öösì)—º;’ˆ¸·¨¨´´```````     \n"
        "  ````````´´´¨·…¸‚›~7Issjƒ•÷¦—÷|)}l‰söö¼}÷~¹‘ˆ…·¨¨´´````````   \n"
        "  ``````´´¨··…˜’°~[c½òì%¡—„““”^÷«×=¢òö¢ƒ}÷³›’¸…·¨¨´````````    \n"
        " ```````´´¨··…’›~¡<ƒ‰s½>[•!÷÷¬«+¿†iíƒuòsöòƒì+„~“¹;’…´´```````  \n"
        " ````´˜’‚’|†1jlíj¤Jƒƒzƒtj=<>iiii>†×¿†77¿)!„–:’¸…·¨´´´```````   \n"
        "  ``````´´¨¨·…ˆ’;­“¯i¼c}>|¯ª””„^¦|)<l¢öì•^°:’¸…·¨¨´````````     \n"
        "   ```````´´¨·…¸˜‘¹“•‡¢‰¢[}!¦¦¦¡•×ƒ‰söc)^–:’¸…·¨¨´````````     \n"
        "    ```````´´´¨¨…¸˜:­„1½¢us½llïosòòss<¡~¹‘ˆ¸·¨´´´´``````        \n"
        "      ````````´´´¨…ˆ’’­”}o‰½¢su¢¢uI»¯º¹’˜¸…¨¨´´```````          \n"
        "        `````````´¨¨·¸ˆ‚‘›²~ª^^„“–¹;‘’¸…·¨´´`````````          \n"
        "          ```````````´¨¨·¸˜˜˜˜˜˜ˆ¸¸…·¨¨´´````````              \n"
        "               ```````````´´´´´´´´´```````````                 \n"
        "                ``´`´´`´``````````````´```                     \n"
        "                   `  `                                        \n"
        "                      ´                                        \n"
    )


def get_centred_art(art: str, width: int) -> str:
    """
    Centre art in a width.

    Args:
        art: The art to be centred.
        width: The number of characters in the region to centre within.

    Returns:
        The art centred within width.
    """
    # Split the line into individual lines
    split_art: List[str] = art.split("\n")

    # Initialise the centred string
    new_string: str = ""

    # Loop over the art string centring each line
    for line in split_art:
        if "+" in line or "|" in line:
            new_string += line.center(width + 4) + "\n"
        else:
            new_string += line.center(width) + "\n"

    return new_string


def print_centred_art(art: str, width: int) -> None:
    """
    Print the art centred in a width.

    Args:
        width: The number of characters in the region to centre within.
    """
    # Get the centred art
    cent_art: str = get_centred_art(art, width)

    print(cent_art)


""
