#!/usr/bin/env python

"""
Install and setup 3rd software which is not contained in distribution's packet management.
All dependencies should use this interface:
* install_<dependency prefix>.py ... specified __name__ == "__main__"
* main_<dependency prefix>() ... entry point to call from global script
* step_<dependency prefix>_<action>() ... private steps to setup and install dependency

The following function should be used for consistence CLI:
* ask_for_execute()
* print_step_begin()
* print_step_end()
"""

__author__ = 'pieloth'

import install_mne
import install_qt5_static


def main():
    print_dependency_header("Install Dependencies")

    if ask_for_execute("Install MNE"):
        install_mne.main_mne()

    print

    if ask_for_execute("Install Qt5"):
        install_qt5_static.main_qt5()

    print


def ask_for_execute(action):
    var = raw_input(action + " y/n? ")
    if var.startswith('y'):
        return True
    else:
        return False


def print_dependency_header(dep_name):
    print(dep_name)
    print('=' * len(dep_name))
    print


def print_step_begin(action_str):
    info = action_str + " ..."
    print(info)
    print('-' * len(info))


def print_step_end(action_str):
    info = action_str + " ... finished!"
    print('-' * len(info))
    print(info)


if __name__ == "__main__":
    main()
