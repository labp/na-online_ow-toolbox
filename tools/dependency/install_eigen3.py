#!/usr/bin/env python

"""
Downloads Eigen library with sparse matrix support.
"""

__author__ = 'pieloth'

import argparse
import os
from subprocess import call

from install import AInstaller
from install import AInstaller as Utils


class Installer(AInstaller):
    REPO_FOLDER = "eigen321"

    def __init__(self, destdir):
        AInstaller.__init__(self, "Eigen", destdir)

    def pre_install(self):
        print("NOTE: You only need this Eigen version, if your version is <3.1.")
        success = True
        success = success and Utils.check_program("hg", "--version")
        success = success and Utils.check_program("ln", "--version")
        return success

    def install(self):
        if Utils.ask_for_execute("Download " + self.NAME):
            self._download()

        print

        if Utils.ask_for_execute("Initialize " + self.NAME):
            self._initialize()

        print

        if Utils.ask_for_execute("Setup " + self.NAME):
            self._setup()

        return True

    def post_install(self):
        print("Before compiling the toolbox, please set the following environment variables:\n")
        include_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        print("    EIGEN3_INCLUDE_DIR=" + include_dir)

        print
        return True

    def _download(self):
        Utils.print_step_begin("Downloading")
        repo = "https://bitbucket.org/eigen/eigen/"
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        call("hg clone " + repo + " " + repo_dir, shell=True)
        Utils.print_step_end("Downloading")

    def _initialize(self):
        Utils.print_step_begin("Initializing")
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        os.chdir(repo_dir)
        version = "3.2.1"
        call("hg update " + version, shell=True)
        Utils.print_step_end("Initializing")

    def _setup(self):
        Utils.print_step_begin("Setup")
        dest_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, "Eigen", "unsupported")
        if not os.path.exists(dest_dir):
            link_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, "unsupported", "Eigen")
            call(["ln", "-s", link_dir, dest_dir])
        Utils.print_step_end("Setup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloads Eigen library with sparse matrix support.")
    parser.add_argument("-d", "--destdir", help="Destination path.")
    args = parser.parse_args()

    destdir = AInstaller.get_default_destdir()
    if args.destdir:
        destdir = args.destdir

    installer = Installer(destdir)
    installer.do_install()
