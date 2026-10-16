import os
import sys

from .runtime.client import native_binary, native_environment

if __name__ == "__main__":
    binary = str(native_binary())
    os.execvpe(binary, [binary, *sys.argv[1:]], native_environment())
