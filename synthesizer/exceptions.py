

class InconsistentParameter(Exception):
    """ Generic exception class for inconsistent parameters.
    """

    def __init__(self, *args):
        """
        Exception raised for inconsistent parameter choice
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{0} '.format(self.message)
        return 'Inconsistent parameter choice'


class InconsistentArguments(Exception):
    """ Generic exception class for inconsistent combinations of arguments.
    """

    def __init__(self, *args):
        """
        Exception raised for inconsistent argument choice
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{0} '.format(self.message)
        return 'Inconsistent parameter choice'


class UnimplementedFunctionality(Exception):
    """ Generic exception class for functionality not yet implemented.
    """

    def __init__(self, *args):
        """
        Exception raised for inconsistent argument choice
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{0} '.format(self.message)
        return 'Unimplemented functionality!'
