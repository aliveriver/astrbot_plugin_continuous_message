class ParseException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class RedirectException(ParseException):
    def __init__(self):
        super().__init__("redirect failed")
