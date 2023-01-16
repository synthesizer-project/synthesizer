

class InconsistentParameter(Exception):
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
        else:
            return 'Inconsistent parameter choice'


class InconsistentAddition(Exception):
    def __init__(self, *args):
        """
        Exception raised for inconsistent adding (a+b=c)
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
        else:
            return 'Unable to add'
