__author__ = 'Christof Pieloth'

import os
from subprocess import call

from packbacker.constants import Parameter
from packbacker.errors import ParameterError
from packbacker.utils import Utils
from packbacker.utils import UtilsUI
from packbacker.installer import Installer


class Pcl(Installer):
    """
    Compiling Point Cloud Library (PCL).
    WWW: http://www.pointclouds.org
    """

    REPO_FOLDER = "pcl"
    BUILD_FOLDER = "build"

    def __init__(self):
        Installer.__init__(self, "pcl", "Point Cloud Library (PCL)")
        self.arg_version = "pcl-1.7.1"  # 2013-10-07

    @classmethod
    def instance(cls, params):
        installer = Pcl()
        if Parameter.DEST_DIR in params:
            installer.arg_dest = params[Parameter.DEST_DIR]
        else:
            raise ParameterError(Parameter.DEST_DIR + ' parameter is missing!')
        if Parameter.VERSION in params:
            installer.arg_version = params[Parameter.VERSION]
        return installer

    @classmethod
    def prototype(cls):
        return Pcl()

    def _pre_install(self):
        UtilsUI.print("NOTE: Before installing PCL from source, please try to install prebuilt binaries:")
        UtilsUI.print("      http://www.pointclouds.org/downloads/")
        success = True
        success = success and Utils.check_program("git", "--version")
        success = success and Utils.check_program("cmake", "--version")
        success = success and Utils.check_program("make", "--version")
        if not Utils.check_program("g++", "--version") and not Utils.check_program("c++", "--version"):
            success = False
        return success

    def _install(self):
        success = True

        if success and UtilsUI.ask_for_execute("Download " + self.name):
            success = success and self.__download()
        if success and UtilsUI.ask_for_execute("Initialize " + self.name):
            success = success and self.__initialize()
        if success and UtilsUI.ask_for_execute("Configure " + self.name):
            success = success and self.__configure()
        if success and UtilsUI.ask_for_execute("Compile " + self.name):
            success = success and self.__compile_install()

        return success

    def _post_install(self):
        pcl_dir = os.path.join(self.arg_dest, self.REPO_FOLDER, self.BUILD_FOLDER)
        UtilsUI.print_env_var("PCL_DIR=", pcl_dir)
        return True

    def __download(self):
        UtilsUI.print_step_begin("Downloading")
        repo = "https://github.com/PointCloudLibrary/pcl.git"
        repo_dir = os.path.join(self.arg_dest, self.REPO_FOLDER)
        call("git clone " + repo + " " + repo_dir, shell=True)
        UtilsUI.print_step_end("Downloading")
        return True

    def __initialize(self):
        UtilsUI.print_step_begin("Initializing")
        repo_dir = os.path.join(self.arg_dest, self.REPO_FOLDER)
        os.chdir(repo_dir)
        call("git checkout " + self.arg_version, shell=True)
        UtilsUI.print_step_end("Initializing")
        return True

    def __configure(self):
        UtilsUI.print_step_begin("Configuring")
        build_dir = os.path.join(self.arg_dest, self.REPO_FOLDER, self.BUILD_FOLDER)
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        else:
            print("You may have to clear the folder:\n" + build_dir)
        os.chdir(build_dir)
        # Check and test, which options can be disabled to get required libs
        # Print dependencies with two following cmake: cmake -D...; cmake -D...
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
        UtilsUI.print_step_end("Configuring")
        return True

    def __compile_install(self):
        UtilsUI.print_step_begin("Compiling & Installing")
        build_dir = os.path.join(self.arg_dest, self.REPO_FOLDER, self.BUILD_FOLDER)
        os.chdir(build_dir)
        jobs = UtilsUI.ask_for_make_jobs()
        call("make -j" + str(jobs), shell=True)
        if UtilsUI.ask_for_execute("Install PCL to system? (requires root/sudo)"):
            call("sudo make install", shell=True)
        UtilsUI.print_step_end("Compiling & Installing")
        return True
