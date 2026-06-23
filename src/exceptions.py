import sys


def error_message_detail(error, error_detail: sys) -> str:
    """Build an error string with file, line, and the active exception if any."""
    _, exc_value, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    message = exc_value if exc_value is not None else error

    return f"Error in [{file_name}] line [{line_no}] message [{message}]"


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    from logger import logging

    try:
        _ = 1 / 0
    except Exception as e:
        logging.exception("Divide by zero error")
        raise CustomException(e, sys) from e
