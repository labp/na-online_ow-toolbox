#!/usr/bin/env python

"""
Setup static Qt5 library and includes for compilation.
"""

__author__ = 'pieloth'

import argparse
import os
from subprocess import call
import sys

from install import AInstaller
from install import AInstaller as Utils


QT5_INSTALL_FOLDER = "qt5_static"


class Installer(AInstaller):
    REPO_FOLDER = "qt5_repository"

    def __init__(self, destdir, installdir):
        AInstaller.__init__(self, "Qt5 Framework", destdir)
        self.INSTALLDIR = installdir

    def pre_install(self):
        success = True
        success = success and Utils.check_program("git", "--version")
        success = success and Utils.check_program("make", "--version")
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

        if Utils.ask_for_execute("Compile & install " + self.NAME):
            self._compile_install()

        return True

    def post_install(self):
        print("Before compiling the toolbox, please set the following environment variables:\n")
        static_root = os.path.join(self.DESTDIR, self.INSTALLDIR)
        print("    QT5_STATIC_ROOT=" + static_root)

        include_dir = os.path.join(self.DESTDIR, self.INSTALLDIR, "include")
        print("    QT5_INCLUDE_DIR=" + include_dir)
        print("    QT5_STATIC_INCLUDE_DIR=" + include_dir)

        static_library_dir = os.path.join(self.DESTDIR, self.INSTALLDIR, "lib")
        print("    QT5_STATIC_LIBRARY_DIR=" + static_library_dir)

        print
        return True

    def _download(self):
        Utils.print_step_begin("Downloading")
        repo = "git://gitorious.org/qt/qt5.git"
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        call("git clone " + repo + " " + repo_dir, shell=True)
        Utils.print_step_end("Downloading")

    def _initialize(self):
        Utils.print_step_begin("Initializing")
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        os.chdir(repo_dir)
        version = "v5.1.1"
        call("git checkout " + version, shell=True)
        qt_module_selection = "--module-subset=qtbase"
        call("./init-repository " + qt_module_selection, shell=True)
        Utils.print_step_end("Initializing")

    def _configure(self):
        Utils.print_step_begin("Configuring")
        install_path = os.path.join(self.DESTDIR, self.INSTALLDIR)
        if not os.path.exists(install_path):
            os.mkdir(install_path)
        else:
            print("You may have to clear the folder:\n" + install_path)
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        os.chdir(repo_dir)
        qt5_configure = "./configure -prefix " + install_path + " -confirm-license -opensource -release -static " \
                                                                "-nomake tests -nomake examples -qt-xcb"
        call(qt5_configure, shell=True)
        Utils.print_step_end("Configuring")

    def _compile_install(self):
        Utils.print_step_begin("Compiling & Installing")
        jobs = Utils.ask_for_make_jobs()
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        os.chdir(repo_dir)
        call("make -j" + str(jobs), shell=True)
        call("make -j" + str(jobs) + " install", shell=True)
        Utils.print_step_end("Compiling & Installing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Installs Qt5 Framework as static library.")
    parser.add_argument("-d", "--destdir", help="Destination path.")
    args = parser.parse_args()

    destdir = AInstaller.get_default_destdir()
    if args.destdir:
        destdir = args.destdir

    installer = Installer(destdir, QT5_INSTALL_FOLDER)
    if installer.do_install():
        sys.exit(AInstaller.EXIT_SUCCESS)
    else:
        sys.exit(AInstaller.EXIT_ERROR)
