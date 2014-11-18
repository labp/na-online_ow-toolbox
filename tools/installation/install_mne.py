#!/usr/bin/env python

"""
Setup MNE-CPP library and includes for compilation.
"""

__author__ = 'pieloth'

import argparse
import os
from subprocess import call
import sys

from install import AInstaller
from install import AInstaller as Utils


class Installer(AInstaller):
    REPO_FOLDER = "mne-cpp"

    def __init__(self, destdir, qt5_root):
        AInstaller.__init__(self, "MNE-CPP", destdir, )
        self.QT5_ROOT = qt5_root

    def pre_install(self):
        success = True
        success = success and Utils.check_program("git", "--version")
        success = success and Utils.check_program("make", "--version")
        qmake5 = os.path.join(self.QT5_ROOT, "bin", "qmake")
        if not Utils.check_program(qmake5, "--version"):
            if not Utils.check_program("qmake5", "--version"):
                print('Please set the alias qmake5, which points to your Qt5 qmake!')
                success = False
        if not Utils.check_program("g++", "--version") and not Utils.check_program("c++", "--version"):
            success = False
        return success

    def install(self):
        if Utils.ask_for_execute("Download " + self.NAME):
            self._download()

        print

        if Utils.ask_for_execute("Initialize " + self.NAME):
            self._initialize()

        print

        if Utils.ask_for_execute("Configure " + self.NAME):
            self._configure()

        print

        if Utils.ask_for_execute("Compile " + self.NAME):
            self._compile()

        return True

    def post_install(self):
        print("Before compiling the toolbox, please set the following environment variables:\n")

        include_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, "MNE")
        print("    MNE_INCLUDE_DIR=" + include_dir)

        library_path = os.path.join(self.DESTDIR, self.REPO_FOLDER, "lib")
        print("    MNE_LIBRARY_DIR=" + library_path)

        print
        return True

    def _download(self):
        Utils.print_step_begin("Downloading")
        repo = "https://github.com/mne-tools/mne-cpp.git"
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        call("git clone " + repo + " " + repo_dir, shell=True)
        Utils.print_step_end("Downloading")

    def _initialize(self):
        Utils.print_step_begin("Initializing")
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        os.chdir(repo_dir)
        version = "b8f6166ca34c01effe5bdc1eedf26dc9aea44899"  # 2014-11-17
        call("git checkout " + version, shell=True)
        Utils.print_step_end("Initializing")

    def _configure(self):
        Utils.print_step_begin("Configuring")
        mne_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, "MNE")
        os.chdir(mne_dir)
        qmake5 = os.path.join(self.QT5_ROOT, "bin", "qmake")
        if not Utils.check_program(qmake5, "--version"):
            qmake5 = "qmake5"
        mne_configure = qmake5 + " -recursive"
        call(mne_configure, shell=True)
        Utils.print_step_end("Configuring")

    def _compile(self):
        Utils.print_step_begin("Compiling")
        mne_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, "MNE")
        os.chdir(mne_dir)
        jobs = AInstaller.ask_for_make_jobs()
        call("make -j" + str(jobs), shell=True)
        Utils.print_step_end("Compiling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Installs MNE-CPP library.")
    parser.add_argument("-d", "--destdir", help="Destination path.")
    parser.add_argument("-q", "--qt5root", help="Path to Qt5 installation.")
    args = parser.parse_args()

    destdir = AInstaller.get_default_destdir()
    if args.destdir:
        destdir = args.destdir

    qt5_root = os.path.join(destdir, "qt5")
    if args.qt5root:
        qt5_root = args.qt5root
    installer = Installer(destdir, qt5_root)
    if installer.do_install():
        sys.exit(AInstaller.EXIT_SUCCESS)
    else:
        sys.exit(AInstaller.EXIT_ERROR)
