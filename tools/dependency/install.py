#!/usr/bin/env python

"""
Install and setup 3rd software which is not contained in distribution's packet management.
All installer must implement AInstaller and check CLI arguments -d and --destdir for destination path.
Furthermore the __main__ must invoke the installation with installer.do_install().
"""

__author__ = 'pieloth'

import argparse
import os
import subprocess
from subprocess import call

DEPENDENCY_PATH_PREFIX = "na-online_dependencies"

class AInstaller:
    """Abstract installer with default implementations of pre_install, install and post_install."""
    
    def __init__(self, name, destdir):
        self.NAME = name
        self.DESTDIR = destdir

    def pre_install(self):
        """Is called before the installation. It can be used to check for tools which are required."""
        return True

    def install(self):
        """Implements the installation."""
        return True

    def post_install(self):
        """Is called after a successful installation. Can be used to test installation or for user instructions."""
        return True

    def do_install(self):
        """Starts the installation process."""
        AInstaller.print_install_begin(self.NAME)

        success = self.pre_install()
        if success:
            success = self.install()

        if success:
            success = self.post_install()

        AInstaller.print_install_end(self.NAME)
        return success

    @staticmethod
    def ask_for_execute(action):
        var = raw_input(action + " y/n? ")
        if var.startswith('y'):
            return True
        else:
            return False


    @staticmethod
    def ask_for_make_jobs():
        jobs = 1
        try:
            jobs = int(raw_input("Number of jobs (recommended: # of CPU cores): "))
        except ValueError:
            print("Wrong input format.")
            jobs = 1
        if jobs < 1:
            jobs = 1
        print("Using job=" + str(jobs))
        return jobs

    @staticmethod
    def print_install_begin(dep_name):
        # print('=' * len(dep_name))
        print('=' * 80)
        print(dep_name)
        print('-' * 80)

    @staticmethod
    def print_install_end(dep_name):
        print('-' * 80)
        print(dep_name)
        print('=' * 80)

    @staticmethod
    def print_step_begin(action_str):
        info = action_str + " ..."
        print(info)
        print('-' * 40)

    @staticmethod
    def print_step_end(action_str):
        info = action_str + " ... finished!"
        print('-' * 40)
        print(info)

    @staticmethod
    def check_program(program, arg):
        try:
            call([program, arg], stdout=subprocess.PIPE)
            return True
        except OSError as e:
            print("Could not found: " + program)
            return False

    @staticmethod
    def get_default_destdir():
        homedir = os.path.expanduser("~")
        destdir = os.path.join(homedir, DEPENDENCY_PATH_PREFIX)
        return destdir
    

class Installer(AInstaller):

    def __init__(self, destdir):
        AInstaller.__init__(self, "Install Dependencies", destdir)

    def install(self):
        destdir_arg = "-d " + self.DESTDIR

        if AInstaller.ask_for_execute("Install Qt5 Framework"):
            call("python install_qt5_static.py " + destdir_arg, shell=True)
            
        print
    
        if AInstaller.ask_for_execute("Install MNE-CPP"):
            call("python install_mne.py " + destdir_arg, shell=True)
    
        print
    
        if AInstaller.ask_for_execute("Install FielTrip Buffer"):
            call("python install_ft_buffer.py " + destdir_arg, shell=True)
        
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Installs 3rd party software and libraries for NA-Online Toolbox.")
    parser.add_argument("-d", "--destdir", help="Destination path.")
    args = parser.parse_args()

    destdir = AInstaller.get_default_destdir()
    if args.destdir:
        destdir = args.destdir

    installer = Installer(destdir)
    installer.do_install()
