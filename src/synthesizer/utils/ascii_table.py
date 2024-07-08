"""A module for formatting objects into ASCII tables.

Contains the TableFormatter class for formatting objects into ASCII tables.

Example usage:
    class Example:
        def __init__(self):
            self.coordinates = np.random.rand(10, 3)
            self.velocities = np.random.rand(10, 3)
            self.masses = np.random.rand(10)
            self.redshift = 0.5
            self.softening_length = 0.1
            self.nparticles = 10
            self.centre = np.array([0.5, 0.5, 0.5])

    # Example usage
    example = Example()
    formatter = TableFormatter(example)
    print(formatter.generate_table())
"""

import numpy as np
from unyt import unyt_array


class TableFormatter:
    """
    A class to format an arbitrary object's attributes into a table.

    Attributes:
        obj (object):
            The object whose attributes are to be formattedinto a table.
        attributes (dict):
            A dictionary of the object's attributes.
    """

    def __init__(self, obj):
        """
        Initialize the TableFormatter with the given object.

        Args:
            obj (object):
                The object to be formatted into a table.
        """
        self.obj = obj
        self.attributes = vars(obj)

    def format_array(self, array):
        """
        Format numpy arrays to show their mean value.

        Args:
            array (numpy.ndarray):
                The array to be formatted.

        Returns:
            str:
                The formatted string showing the mean value of the array.
        """
        return (
            f"{np.min(array):.2e} -> {np.max(array):.2e} "
            f"(Mean: {np.mean(array):.2e})"
        )

    def format_dict(self, dictionary):
        """
        Format dictionaries to show their keys and the types of their values.

        Args:
            dictionary (dict):
                The dictionary to be formatted.

        Returns:
            list of tuple:
                A list of tuples where each tuple contains the dictionary
                key and the type of its value.
        """
        return [
            (key, type(value).__name__) for key, value in dictionary.items()
        ]

    def get_value_rows(self):
        """
        Collect the object's attributes and formats them into rows.

        Returns:
            list of tuple:
                A list of tuples where each tuple contains the attribute
                name and its formatted value.
        """
        rows = []
        for attr, value in self.attributes.items():
            if value is None:
                continue
            # Handle Quantitys
            if attr[0] == "_" and hasattr(self.obj, attr[1:]):
                attr = attr[1:]
                value = getattr(self.obj, attr)
            if not (
                isinstance(value, dict) or isinstance(value, np.ndarray)
            ) or (isinstance(value, np.ndarray) and value.size <= 3):
                # Handle the different situations
                if isinstance(value, float) and (value >= 1e4 or value < 0.01):
                    formatted_value = f"{value:.2e}"
                elif isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                elif isinstance(value, int):
                    formatted_value = str(value)
                elif isinstance(value, unyt_array):
                    formatted_value = f"{value.value:.2e} {value.units}"
                elif hasattr(value, "__str__"):
                    formatted_value = value.__repr__()
                else:
                    formatted_value = str(value)
                rows.append((attr, formatted_value))
        return rows

    def get_array_rows(self):
        """
        Collect the object's attributes and formats them into rows.

        Returns:
            list of tuple:
                A list of tuples where each tuple contains the attribute
                name and its formatted value.
        """
        rows = []
        for attr, value in self.attributes.items():
            # Handle Quantitys
            if attr[0] == "_" and hasattr(self.obj, attr[1:]):
                attr = attr[1:]
                value = getattr(self.obj, attr)
            if isinstance(value, (np.ndarray, unyt_array)) and value.size > 3:
                formatted_value = self.format_array(value)
                rows.append((f"{attr} {value.shape}", formatted_value))
        return rows

    def get_dict_rows(self):
        """
        Collect the object's attributes and formats them into rows.

        Returns:
            list of tuple:
                A list of tuples where each tuple contains the attribute
                name and its formatted value.
        """
        rows = []
        for attr, value in self.attributes.items():
            if isinstance(value, dict):
                formatted_values = self.format_dict(value)
                for i, (key, formatted_value) in enumerate(formatted_values):
                    if i == 0:
                        rows.append((attr, f"{key}: {formatted_value}"))
                    else:
                        rows.append(("", f"{key}: {formatted_value}"))
        return rows

    def get_table(self, title_text):
        """
        Generate a formatted table.

        Args:
            title_text (str):
                The text to be displayed above the table. This will be upper
                cased.

        Returns:
            str:
                The formatted table as a string.
        """
        rows = self.get_value_rows()
        rows.extend(self.get_array_rows())
        rows.extend(self.get_dict_rows())
        col_widths = [
            max(len(str(item)) for item in col)
            for col in zip(*rows, ("Property", "Value"))
        ]

        def format_row(row):
            return f"| {row[0]:<{col_widths[0]}} | {row[1]:<{col_widths[1]}} |"

        header = format_row(("Attribute", "Value"))
        separator = (
            f"+{'-' * (col_widths[0] + 2)}+{'-' * (col_widths[1] + 2)}+"
        )

        # Define the title
        title = (
            f"| {title_text:^{col_widths[0] + col_widths[1] + 3}} |".upper()
        )

        lines = [
            "+" + "-" * (col_widths[0] + col_widths[1] + 5) + "+",
            title,
            separator,
            header,
        ]
        for row in rows:
            if row[1] == "None":
                continue
            if row[0] != "":
                lines.append(separator)
            lines.append(format_row(row))

        # Finish the table
        lines.append(separator)

        return "\n".join(lines)
