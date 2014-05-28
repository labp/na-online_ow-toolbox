#!/usr/bin/env python

"""
Setup FieldTrip Buffer library and includes for compilation.
"""

__author__ = 'pieloth'

import argparse
import os
from subprocess import call
import sys

from install import AInstaller
from install import AInstaller as Utils


class Installer(AInstaller):
    REPO_FOLDER = "fieldtrip"
    FTB_BUFFER_INCLUDE = "realtime/src/buffer/src"
    FTB_BUFFER_LIBRARY = "libFtbBuffer.a"
    FTB_CLIENT_INCLUDE = "realtime/src/buffer/cpp"
    FTB_CLIENT_LIBRARY = "libFtbClient.a"

    def __init__(self, destdir):
        AInstaller.__init__(self, "FieldTrip Buffer", destdir, )

    def pre_install(self):
        success = True
        success = success and Utils.check_program("git", "--version")
        success = success and Utils.check_program("make", "--version")
        success = success and Utils.check_program("gcc", "--version")
        success = success and Utils.check_program("g++", "--version")
        return success

    def install(self):
        if Utils.ask_for_execute("Download " + self.NAME):
            self._download()

        print

        if Utils.ask_for_execute("Initialize " + self.NAME):
            self._initialize()

        print

        if Utils.ask_for_execute("Compile " + self.NAME):
            self._compile()

        return True

    def post_install(self):
        print("Before compiling the toolbox, please set the following environment variables:\n")
        ftb_buffer_include_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, self.FTB_BUFFER_INCLUDE)
        print("    FTB_BUFFER_INCLUDE_DIR=" + ftb_buffer_include_dir)

        ftb_buffer_lib = os.path.join(self.DESTDIR, self.REPO_FOLDER, self.FTB_BUFFER_INCLUDE,
                                      self.FTB_BUFFER_LIBRARY)
        print("    FTB_BUFFER_LIBRARY=" + ftb_buffer_lib)

        ftb_client_include_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, self.FTB_CLIENT_INCLUDE)
        print("    FTB_CLIENT_INCLUDE_DIR=" + ftb_client_include_dir)

        ftb_client_lib = os.path.join(self.DESTDIR, self.REPO_FOLDER, self.FTB_CLIENT_INCLUDE,
                                      self.FTB_CLIENT_LIBRARY)
        print("    FTB_CLIENT_LIBRARY=" + ftb_client_lib)

        print
        return True

    def _download(self):
        Utils.print_step_begin("Downloading")
        # repo = "~/workspace/fieldtrip" # clone from local repository
        repo = "https://github.com/fieldtrip/fieldtrip.git"
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        call("git clone " + repo + " " + repo_dir, shell=True)
        Utils.print_step_end("Downloading")

    def _initialize(self):
        Utils.print_step_begin("Initializing")
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        os.chdir(repo_dir)
        version = "de8b915fd8376549aad3c27f1086090dfa0d0071"  # 2014-05-02
        call("git checkout " + version, shell=True)
        Utils.print_step_end("Initializing")

    def _compile(self):
        Utils.print_step_begin("Compiling")
        self._compile_ftb_buffer()
        self._compile_ftb_client()
        Utils.print_step_end("Compiling")

    def _compile_ftb_buffer(self):
        buffer_path = os.path.join(self.DESTDIR, self.REPO_FOLDER, self.FTB_BUFFER_INCLUDE)
        os.chdir(buffer_path)
        call("make -j2", shell=True)
        call("cp libbuffer.a " + self.FTB_BUFFER_LIBRARY, shell=True)

    def _compile_ftb_client(self):
        client_path = os.path.join(self.DESTDIR, self.REPO_FOLDER, self.FTB_CLIENT_INCLUDE)
        os.chdir(client_path)
        call("g++ -c FtConnection.cc -I../src -I. -Wunused -Wall -pedantic -O3 -fPIC", shell=True)
        call("ar rv " + self.FTB_CLIENT_LIBRARY + " FtConnection.o", shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Installs FieldTrip Buffer for real-time streaming.")
    parser.add_argument("-d", "--destdir", help="Destination path.")
    args = parser.parse_args()

    destdir = AInstaller.get_default_destdir()
    if args.destdir:
        destdir = args.destdir

    installer = Installer(destdir)
    if installer.do_install():
        sys.exit(AInstaller.EXIT_SUCCESS)
    else:
        sys.exit(AInstaller.EXIT_ERROR)
