#!/usr/bin/env python

"""
Setup FieldTrip Buffer library and includes for compilation.
"""

__author__ = 'pieloth'

from subprocess import call
import os

import install_dependencies as deputil

_FT_DESTDIR = "/tmp"
_FT_REPO_FOLDER = "fieldtrip"
_FTB_BUFFER_INCLUDE = "realtime/src/buffer/src"
_FTB_BUFFER_LIBRARY = "libFtbBuffer.a"
_FTB_CLIENT_INCLUDE = "realtime/src/buffer/cpp"
_FTB_CLIENT_LIBRARY = "libFtbClient.a"

def main_ft():
    deputil.print_dependency_header("Install FieldTrip Buffer")

    if deputil.ask_for_execute("Download FieldTrip"):
        step_ft_download_repo()

    print
    
    if deputil.ask_for_execute("Initialize FielTrip repository"):
        step_ft_init_repo()

    print

    if deputil.ask_for_execute("Compile FieldTrip"):
        step_ft_compile_install()

    print

    print("Before compiling the toolbox, please set the following environment variables:\n")
    print("\tFTB_BUFFER_INCLUDE_DIR=" + _FT_DESTDIR + "/" + _FT_REPO_FOLDER + "/" + _FTB_BUFFER_INCLUDE)
    ftb_buffer_lib = _FT_DESTDIR + "/" + _FT_REPO_FOLDER + "/" + _FTB_BUFFER_INCLUDE + "/" + _FTB_BUFFER_LIBRARY
    print("\tFTB_BUFFER_LIBRARY=" + ftb_buffer_lib)
    print("\tFTB_CLIENT_INCLUDE_DIR=" + _FT_DESTDIR + "/" + _FT_REPO_FOLDER + "/" + _FTB_CLIENT_INCLUDE)
    ftb_client_lib = _FT_DESTDIR + "/" + _FT_REPO_FOLDER + "/" + _FTB_CLIENT_INCLUDE + "/" + _FTB_CLIENT_LIBRARY
    print("\tFTB_CLIENT_LIBRARY=" + ftb_client_lib)


def step_ft_download_repo():
    deputil.print_step_begin("Downloading")

    # repo = "~/workspace/fieldtrip" # clone from local repository
    repo = "https://github.com/fieldtrip/fieldtrip.git"
    call("git clone " + repo + " " + _FT_DESTDIR + "/" + _FT_REPO_FOLDER, shell=True)

    deputil.print_step_end("Downloading")


def step_ft_init_repo():
    deputil.print_step_begin("Initializing")

    os.chdir(_FT_DESTDIR + "/" + _FT_REPO_FOLDER)
    version = "de8b915fd8376549aad3c27f1086090dfa0d0071" # 2014-05-02
    call("git checkout " + version, shell=True)

    deputil.print_step_end("Initializing")


def step_ft_compile_install():
    deputil.print_step_begin("Compiling")

    _ft_compile_ftb_buffer()
    _ft_compile_ftb_client()

    deputil.print_step_end("Compiling")
    
    
def _ft_compile_ftb_buffer():
    os.chdir(_FT_DESTDIR + "/" + _FT_REPO_FOLDER + "/" + _FTB_BUFFER_INCLUDE)
    call("make -j2", shell=True)
    call("cp libbuffer.a " + _FTB_BUFFER_LIBRARY, shell=True)


def _ft_compile_ftb_client():
    os.chdir(_FT_DESTDIR + "/" + _FT_REPO_FOLDER + "/" + _FTB_CLIENT_INCLUDE)
    # call("make -j2", shell=True) # we need -fPIC
    call("g++ -c FtConnection.cc -I../src -I. -Wunused -Wall -pedantic -O3 -fPIC", shell=True)
    call("ar rv " + _FTB_CLIENT_LIBRARY + " FtConnection.o", shell=True)


if __name__ == "__main__":
    main_ft()
