import os, sys
root = "src".join(os.path.dirname(os.path.abspath(__file__)).split("src")[:-1]).strip("/").strip("\\")
print(f"Library root directory: {root}")