import sys

from visionguard.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "predict")
    app()
