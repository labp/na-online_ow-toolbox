//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
// For more information see http://www.openwalnut.org/copying
//
// This file is part of OpenWalnut.
//
// OpenWalnut is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// OpenWalnut is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with OpenWalnut. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#ifndef WLIOSTATUS_H_
#define WLIOSTATUS_H_

#include <string>

/**
 * Return Codes and type for I/O objects, e.g. reader and writer.
 *
 * @author pieloth
 */
namespace WLIOStatus
{
    typedef char ioStatus_t;

    const ioStatus_t _USER_OFFSET = 64; /**< Offset for user defined (reader/writer) status codes. */

    const ioStatus_t SUCCESS = 0;
    const ioStatus_t ERROR_UNKNOWN = 1; /**< Unknown error */
    const ioStatus_t ERROR_FOPEN = 2; /**< Error opening file */
    const ioStatus_t ERROR_FREAD = 3; /**< File read error */
    const ioStatus_t ERROR_FWRITE = 4; /**< File write error */

    /**
     * Returns a description for a status code.
     *
     * @param statusCode
     *
     * @return Description for status code.
     */
    std::string description( ioStatus_t statusCode );

    /**
     * Provides an interface for user-defined status codes, e.g. for reader and writer.
     */
    class WLIOStatusInterpreter
    {
    public:
        virtual ~WLIOStatusInterpreter();

        /**
         * Returns a description for a status code.
         * Default implementation wraps WLIOStatus::description().
         *
         * @param status
         *
         * @return Description for status code.
         */
        virtual std::string getIOStatusDescription( WLIOStatus::ioStatus_t status );

    protected:
        WLIOStatusInterpreter();
    };
}

#endif  // WLIOSTATUS_H_