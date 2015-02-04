__author__ = 'Christof Pieloth'

import os
from subprocess import call

from packbacker.constants import Parameter
from packbacker.errors import ParameterError
from packbacker.utils import Utils
from packbacker.utils import UtilsUI
from packbacker.installer import Installer


class FtBuffer(Installer):
    """
    Setup FieldTrip Buffer library and includes for compilation.
    WWW: fieldtrip.fcdonders.nl/development/realtime
    """
    REPO_FOLDER = "fieldtrip"
    FTB_BUFFER_INCLUDE = "realtime/src/buffer/src"
    FTB_BUFFER_LIBRARY = "libFtbBuffer.a"
    FTB_CLIENT_INCLUDE = "realtime/src/buffer/cpp"
    FTB_CLIENT_LIBRARY = "libFtbClient.a"

    def __init__(self):
        Installer.__init__(self, 'ftbuffer', 'FieldTrip Buffer')

    @classmethod
    def instance(cls, params):
        installer = FtBuffer()
        if Parameter.DEST_DIR in params:
            installer.arg_dest = params[Parameter.DEST_DIR]
        else:
            raise ParameterError(Parameter.DEST_DIR + ' parameter is missing!')
        return installer

    @classmethod
    def prototype(cls):
        return FtBuffer()

    def _pre_install(self):
        success = True
        success = success and Utils.check_program("git", "--version")
        success = success and Utils.check_program("make", "--version")
        success = success and Utils.check_program("gcc", "--version")
        success = success and Utils.check_program("g++", "--version")
        return success

    def _install(self):
        success = True

        if success and UtilsUI.ask_for_execute("Download " + self.name):
            success = success and self.__download()
        if success and UtilsUI.ask_for_execute("Initialize " + self.name):
            success = success and self.__initialize()
        if success and UtilsUI.ask_for_execute("Compile " + self.name):
            success = success and self.__compile()

        return success

    def _post_install(self):
        ftb_buffer_include_dir = os.path.join(self.arg_dest, self.REPO_FOLDER, self.FTB_BUFFER_INCLUDE)
        UtilsUI.print_env_var("FTB_BUFFER_INCLUDE_DIR=", ftb_buffer_include_dir)

        ftb_buffer_lib = os.path.join(self.arg_dest, self.REPO_FOLDER, self.FTB_BUFFER_INCLUDE,
                                      self.FTB_BUFFER_LIBRARY)
        UtilsUI.print_env_var("FTB_BUFFER_LIBRARY=", ftb_buffer_lib)

        ftb_client_include_dir = os.path.join(self.arg_dest, self.REPO_FOLDER, self.FTB_CLIENT_INCLUDE)
        UtilsUI.print_env_var("FTB_CLIENT_INCLUDE_DIR=", ftb_client_include_dir)

        ftb_client_lib = os.path.join(self.arg_dest, self.REPO_FOLDER, self.FTB_CLIENT_INCLUDE,
                                      self.FTB_CLIENT_LIBRARY)
        UtilsUI.print_env_var("FTB_CLIENT_LIBRARY=", ftb_client_lib)
        return True

    def __download(self):
        UtilsUI.print_step_begin("Downloading")
        repo = "https://github.com/fieldtrip/fieldtrip.git"
        repo_dir = os.path.join(self.arg_dest, self.REPO_FOLDER)
        call("git clone " + repo + " " + repo_dir, shell=True)
        UtilsUI.print_step_end("Downloading")
        return True

    def __initialize(self):
        UtilsUI.print_step_begin("Initializing")
        repo_dir = os.path.join(self.arg_dest, self.REPO_FOLDER)
        os.chdir(repo_dir)
        version = "bc7d5c3c0c9a52bc4e15c892deb87fffabe76890"  # 2014-11-19
        call("git checkout " + version, shell=True)
        UtilsUI.print_step_end("Initializing")
        return True

    def __compile(self):
        UtilsUI.print_step_begin("Compiling")
        self.__compile_ftb_buffer()
        self.__compile_ftb_client()
        UtilsUI.print_step_end("Compiling")
        return True

    def __compile_ftb_buffer(self):
        buffer_path = os.path.join(self.arg_dest, self.REPO_FOLDER, self.FTB_BUFFER_INCLUDE)
        os.chdir(buffer_path)
        call("make -j2", shell=True)
        call("cp libbuffer.a " + self.FTB_BUFFER_LIBRARY, shell=True)
        return True

    def __compile_ftb_client(self):
        client_path = os.path.join(self.arg_dest, self.REPO_FOLDER, self.FTB_CLIENT_INCLUDE)
        os.chdir(client_path)
        call("g++ -c FtConnection.cc -I../src -I. -Wunused -Wall -pedantic -O3 -fPIC", shell=True)
        call("ar rv " + self.FTB_CLIENT_LIBRARY + " FtConnection.o", shell=True)
        return True
