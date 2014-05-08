#!/usr/bin/env python

"""
Setup MNE-CPP library and includes for compilation.
"""

__author__ = 'pieloth'

from subprocess import call
import os

import install_dependencies as deputil
import install_qt5_static as qt5

_MNE_DESTDIR = "/tmp"
_MNE_REPO_FOLDER = "mne-cpp"
_MNE_QMAKE5 = qt5.QT5_DESTDIR + "/" + qt5.QT5_INSTALL_FOLDER + "/bin/qmake"

def main_mne():
    deputil.print_dependency_header("Install MNE")

    if deputil.ask_for_execute("Download MNE"):
        step_mne_download_repo()

    print
    
    if deputil.ask_for_execute("Initialize MNE repository"):
        step_mne_init_repo()
        
    print

    if deputil.ask_for_execute("Configure MNE"):
        step_mne_configure()

    print

    if deputil.ask_for_execute("Compile MNE"):
        step_mne_compile_install()

    print

    print("Before compiling the toolbox, please set the following environment variables:\n")
    print("\tMNE_INCLUDE_DIR=" + _MNE_DESTDIR + "/" + _MNE_REPO_FOLDER + "/MNE")
    print("\tMNE_LIBRARY_DIR=" + _MNE_DESTDIR + "/" + _MNE_REPO_FOLDER + "/lib")


def step_mne_download_repo():
    deputil.print_step_begin("Downloading")

    mne_repo = "https://github.com/mne-tools/mne-cpp.git"
    call("git clone " + mne_repo + " " + _MNE_DESTDIR + "/" + _MNE_REPO_FOLDER, shell=True)

    deputil.print_step_end("Downloading")

def step_mne_init_repo():
    deputil.print_step_begin("Initializing")

    os.chdir(_MNE_DESTDIR + "/" + _MNE_REPO_FOLDER)
    version = "140f19b51738719db5d66c5a5259ae3e5c759cac" # 2014-05-08
    call("git checkout " + version, shell=True)

    deputil.print_step_end("Initializing")
    

def step_mne_configure():
    deputil.print_step_begin("Configuring")

    os.chdir(_MNE_DESTDIR + "/" + _MNE_REPO_FOLDER + "/MNE")
    mne_configure = _MNE_QMAKE5 + " -recursive"
    call(mne_configure, shell=True)

    deputil.print_step_end("Configuring")


def step_mne_compile_install():
    deputil.print_step_begin("Compiling")

    os.chdir(_MNE_DESTDIR + "/" + _MNE_REPO_FOLDER + "/MNE")
    try:
        jobs = int(raw_input("Number of jobs (recommended: # of CPU cores): "))
    except ValueError:
        jobs = 1
    print("Using job=" + str(jobs))

    call("make -j" + str(jobs), shell=True)

    deputil.print_step_end("Compiling")


if __name__ == "__main__":
    main_mne()
