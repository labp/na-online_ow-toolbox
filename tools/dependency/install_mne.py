#!/usr/bin/env python

"""
Setup MNE-CPP library and includes for compilation.
"""

__author__ = 'pieloth'

from subprocess import call
import os

from install import AInstaller
from install import AInstaller as Utils

MNE_DESTDIR = "/tmp"
MNE_QT5_ROOT = os.path.join(MNE_DESTDIR, "qt5_static")


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
        success = success and Utils.check_program(qmake5, "--version")
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

    def post_install(self):
        print("Before compiling the toolbox, please set the following environment variables:\n")

        include_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, "MNE")
        print("\tMNE_INCLUDE_DIR=" + include_dir)

        library_path = os.path.join(self.DESTDIR, self.REPO_FOLDER, "lib")
        print("\tMNE_LIBRARY_DIR=" + library_path)

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
        version = "140f19b51738719db5d66c5a5259ae3e5c759cac" # 2014-05-08
        call("git checkout " + version, shell=True)
        Utils.print_step_end("Initializing")

    def _configure(self):
        Utils.print_step_begin("Configuring")
        mne_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, "MNE")
        os.chdir(mne_dir)
        qmake5 = os.path.join(self.QT5_ROOT, "bin", "qmake")
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
    installer = Installer("/tmp", MNE_QT5_ROOT)
    installer.do_install()
