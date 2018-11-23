#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: log.py

## Copyright (c) 2010-2017 Thomas Martin <thomas.martin.1@ulaval.ca>
## 
## This file is part of ORB
##
## ORB is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ORB is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORB.  If not, see <http://www.gnu.org/licenses/>.

    
def setup_socket_logging():
    import logging
    import logging.handlers
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    socketHandler = logging.handlers.SocketHandler(
        'localhost',logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    rootLogger.addHandler(socketHandler)


def print_caller_traceback(self):
    """Print the traceback of the calling function."""
    import logging

    traceback = inspect.stack()
    traceback_msg = ''
    for i in range(len(traceback))[::-1]:
        traceback_msg += ('  File %s'%traceback[i][1]
                          + ', line %d'%traceback[i][2]
                          + ', in %s\n'%traceback[i][3] +
                          traceback[i][4][0])

    logging.debug('\r' + traceback_msg)

