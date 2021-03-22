#!/usr/bin/env python
print("execution will start soon...")
import subprocess
with open("output.txt", "w+") as output:
    subprocess.call(["python3", "./Code_1a/main_Simon.py"], stdout=output);
("finished execution")
