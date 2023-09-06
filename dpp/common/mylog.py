import datetime
import sys


class Logger:
    @classmethod
    def debug(cls, msg):
        ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
        theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        path = sys._getframe(1).f_code.co_filename.split("/")[-1]
        line = sys._getframe(1).f_lineno
        print(
            "\033[1;35m [DEBUG]--> {}:{} {} {} \033[0m".format(path, line, theTime, msg))

    @classmethod
    def info(cls, msg):
        ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
        theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        print("\033[1;34m [INFO]--> {}:{} \033[0m".format(theTime, msg))

    @classmethod
    def error(cls, msg):
        ISOTIMEFORMAT = '%Y-%m-%d %H:%M'
        theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        path = sys._getframe(1).f_code.co_filename.split("/")[-1]
        line = sys._getframe(1).f_lineno
        print(
            "\033[1;41m [ERROR]--> {}:{} {} {} \033[0m".format(path, line, theTime, msg))

    @classmethod
    def warning(cls, msg):
        ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
        theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        path = sys._getframe(1).f_code.co_filename.split("/")[-1]
        line = sys._getframe(1).f_lineno
        print(
            "\033[1;33m [WARNING]--> {}:{} {} {} \033[0m".format(path, line, theTime, msg))
