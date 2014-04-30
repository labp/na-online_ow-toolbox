#!/usr/bin/env python

"""
Setup static Qt5 library and includes for compilation.
"""

__author__ = 'pieloth'

from subprocess import call
import os

import install_dependencies as deputil

QT5_DESTDIR = "/tmp"
_QT5_REPO_FOLDER = "qt5_repository"
QT5_INSTALL_FOLDER = "qt5_static"


def main_qt5():
    deputil.print_dependency_header("Install static Qt5")

    if deputil.ask_for_execute("Download Qt5"):
        step_qt5_download_repo()

    print

    if deputil.ask_for_execute("Initialize Qt5 repository"):
        step_qt5_init_repo()

    print

    if deputil.ask_for_execute("Configure Qt5"):
        step_qt5_configure()

    print

    if deputil.ask_for_execute("Compile & install Qt5"):
        step_qt5_compile_install()

    print

    print("Before compiling the toolbox, please set the following environment variables:\n")
    print("\tQT5_STATIC_ROOT=" + QT5_DESTDIR + "/" + QT5_INSTALL_FOLDER)
    print("\tQT5_INCLUDE_DIR=" + QT5_DESTDIR + "/" + QT5_INSTALL_FOLDER + "/include")
    print("\tQT5_STATIC_INCLUDE_DIR=" + QT5_DESTDIR + "/" + QT5_INSTALL_FOLDER + "/include")
    print("\tQT5_STATIC_LIBRARY_DIR=" + QT5_DESTDIR + "/" + QT5_INSTALL_FOLDER + "/lib")


def step_qt5_download_repo():
    deputil.print_step_begin("Downloading")

    qt5_repo = "git://gitorious.org/qt/qt5.git"
    call("git clone " + qt5_repo + " " + QT5_DESTDIR + "/" + _QT5_REPO_FOLDER, shell=True)

    deputil.print_step_end("Downloading")


def step_qt5_init_repo():
    deputil.print_step_begin("Initializing")

    os.chdir(QT5_DESTDIR + "/" + _QT5_REPO_FOLDER)
    qt_version = "v5.1.1"
    call("git checkout " + qt_version, shell=True)
    qt_module_selection = "--module-subset=qtbase"
    call("./init-repository " + qt_module_selection, shell=True)

    deputil.print_step_end("Initializing")


def step_qt5_configure():
    deputil.print_step_begin("Configuring")

    qt_install_path = QT5_DESTDIR + "/" + QT5_INSTALL_FOLDER
    call("mkdir " + qt_install_path, shell=True)
    os.chdir(QT5_DESTDIR + "/" + _QT5_REPO_FOLDER)
    qt5_configure = "./configure -prefix " + qt_install_path + " -confirm-license -opensource -release -static " \
                                                               "-nomake tests -nomake examples -qt-xcb"
    call(qt5_configure, shell=True)

    deputil.print_step_end("Configuring")


def step_qt5_compile_install():
    deputil.print_step_begin("Compiling & Installing")

    os.chdir(QT5_DESTDIR + "/" + _QT5_REPO_FOLDER)
    try:
        jobs = int(raw_input("Number of jobs (recommended: # of CPU cores): "))
    except ValueError:
        jobs = 1
    print("Using job=" + str(jobs))

    call("make -j" + str(jobs) + " install", shell=True)

    deputil.print_step_end("Compiling & Installing")


if __name__ == "__main__":
    main_qt5()
