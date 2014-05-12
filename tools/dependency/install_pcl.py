#!/usr/bin/env python

"""
Compiling Point Cloud Library (PCL).
WWW: http://www.pointclouds.org
"""

__author__ = 'pieloth'

import argparse
import os
from subprocess import call

from install import AInstaller
from install import AInstaller as Utils


class Installer(AInstaller):
    REPO_FOLDER = "pcl"
    BUILD_FOLDER = "build"

    def __init__(self, destdir):
        AInstaller.__init__(self, "Point Cloud Library (PCL)", destdir, )

    def pre_install(self):
        print("NOTE: Before installing PCL from source, please try to install prebuilt binaries:")
        print("      http://www.pointclouds.org/downloads/")
        success = True
        success = success and Utils.check_program("git", "--version")
        success = success and Utils.check_program("cmake", "--version")
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

        if Utils.ask_for_execute("Compile " + self.NAME):
            self._compile_install()

        return True

    def post_install(self):
        return True

    def _download(self):
        Utils.print_step_begin("Downloading")
        repo = "https://github.com/PointCloudLibrary/pcl.git"
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        call("git clone " + repo + " " + repo_dir, shell=True)
        Utils.print_step_end("Downloading")

    def _initialize(self):
        Utils.print_step_begin("Initializing")
        repo_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER)
        os.chdir(repo_dir)
        version = "pcl-1.7.1"  # 2013-10-07
        call("git checkout " + version, shell=True)
        Utils.print_step_end("Initializing")

    def _configure(self):
        Utils.print_step_begin("Configuring")
        build_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, self.BUILD_FOLDER)
        os.mkdir(build_dir)
        os.chdir(build_dir)
        # Check and test, which options can be disabled to get required libs
        # Print dependencies with two following cmake: cmake -D...; cmake -D...
        # NA-Online requires: common, kdtree, registration, search
        options = []
        options.append("-DCMAKE_BUILD_TYPE=Release")
        options.append("-DBUILD_apps=OFF")
        options.append("-DBUILD_examples=OFF")
        options.append("-DBUILD_geometry=OFF")
        options.append("-DBUILD_global_tests=OFF")
        options.append("-DBUILD_io=OFF")
        options.append("-DBUILD_segmentation=OFF")
        options.append("-DBUILD_surface=OFF")
        options.append("-DBUILD_surface_on_nurbs=OFF")
        options.append("-DBUILD_tracking=OFF")
        options.append("-DBUILD_visualization=OFF")
        cmake_cmd = "cmake " + ' '.join(options) + " ../"
        call(cmake_cmd, shell=True)
        Utils.print_step_end("Configuring")

    def _compile_install(self):
        Utils.print_step_begin("Compiling & Installing")
        build_dir = os.path.join(self.DESTDIR, self.REPO_FOLDER, self.BUILD_FOLDER)
        os.chdir(build_dir)
        jobs = AInstaller.ask_for_make_jobs()
        call("make -j" + str(jobs), shell=True)
        if Utils.ask_for_execute("Install PCL to system? (requires root/sudo)"):
            call("sudo make install")
        Utils.print_step_end("Compiling & Installing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compiles Point Cloud Library (PCL).")
    parser.add_argument("-d", "--destdir", help="Destination path.")
    args = parser.parse_args()

    destdir = AInstaller.get_default_destdir()
    if args.destdir:
        destdir = args.destdir

    installer = Installer(destdir)
    installer.do_install()
