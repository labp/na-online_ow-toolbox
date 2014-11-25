"""
Install and setup 3rd software which is not contained in distribution's packet management.
All installers must implement AInstaller and check CLI arguments -d and --destdir for destination path.
Furthermore the __main__ must invoke the installation with installers.do_install().
"""

__author__ = 'Christof Pieloth'

import logging
import os

from packbacker.utils import UtilsUI


class Installer(object):
    """Abstract installers with default implementations of pre_install and post_install."""

    def __init__(self, name, label):
        self._name = name
        self._label = label
        self._arg_dest = os.path.expanduser('~')
        self._log = logging.getLogger(self._name)

    @property
    def name(self):
        """Short name of the installers."""
        return self._name

    @property
    def label(self):
        """Long name of the installers."""
        return self._label

    @property
    def arg_dest(self):
        """Destination directory."""
        return self._arg_dest

    @arg_dest.setter
    def arg_dest(self, dest):
        self._arg_dest = os.path.expanduser(dest)

    @property
    def log(self):
        """Logger for this installers."""
        return self._log

    def _pre_install(self):
        """Is called before the installation. It can be used to check for tools which are required."""
        return True

    def _install(self):
        """Abstract method, implements the installation."""
        self.log.debug('No yet implemented: ' + str(self.name))
        return False

    def _post_install(self):
        """Is called after a successful installation. Can be used to test installation or for user instructions."""
        return True

    def install(self):
        """Starts the installation process."""
        UtilsUI.print_install_begin(self.label)

        try:
            success = self._pre_install()
            if success:
                success = self._install()

            if success:
                success = self._post_install()
        except Exception as ex:
            success = False
            self.log.error("Unexpected error:\n" + str(ex))

        UtilsUI.print_install_end(self.label)
        return success

    @classmethod
    def instance(cls, params):
        """
        Abstract method, returns an initialized instance of a specific command.
        Can throw a ParameterError, if parameters are missing.
        """
        raise Exception('Instance method not implemented for: ' + str(cls))

    @classmethod
    def prototype(cls):
        """Abstract method, returns an instance of a specific command, e.g. for matches() or is_available()"""
        raise Exception('Prototype method not implemented for: ' + str(cls))

    def matches(self, installer):
        """Checks if this command should be used for execution."""
        return installer.lower().startswith(self.name)