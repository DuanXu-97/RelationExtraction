from __future__ import division, print_function, unicode_literals

import sys
import os.path
from torch import nn


def fwrite(new_doc, path, mode='w'):
    with open(path, mode) as f:
        f.write(new_doc)


def shell(cmd, working_directory='.', stdout=False, stderr=False):
    import subprocess
    from subprocess import PIPE, Popen

    subp = Popen(cmd, shell=True, stdout=PIPE,
                 stderr=subprocess.STDOUT, cwd=working_directory)
    subp_stdout, subp_stderr = subp.communicate()

    if subp_stdout: subp_stdout = subp_stdout.decode("utf-8")
    if subp_stderr: subp_stderr = subp_stderr.decode("utf-8")

    if stdout and subp_stdout:
        print("[stdout]", subp_stdout, "[end]")
    if stderr and subp_stderr:
        print("[stderr]", subp_stderr, "[end]")

    return subp_stdout, subp_stderr

