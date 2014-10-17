//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#ifndef WLIOSTATUS_H_
#define WLIOSTATUS_H_

#include <string>

/**
 * Return Codes and type for I/O objects, e.g. reader and writer.
 *
 * \author pieloth
 */
namespace WLIOStatus
{
    typedef char IOStatusT;

    const IOStatusT _USER_OFFSET = 64; /**< Offset for user defined (reader/writer) status codes. */

    const IOStatusT SUCCESS = 0; /**< I/O operation successful. */
    const IOStatusT ERROR_UNKNOWN = 1; /**< Unknown error */
    const IOStatusT ERROR_FOPEN = 2; /**< Error opening file */
    const IOStatusT ERROR_FREAD = 3; /**< File read error */
    const IOStatusT ERROR_FWRITE = 4; /**< File write error */

    /**
     * Returns a description for a status code.
     *
     * \param statusCode
     *
     * \return Description for status code.
     */
    std::string description( IOStatusT statusCode );

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
         * \param status
         *
         * \return Description for status code.
         */
        virtual std::string getIOStatusDescription( WLIOStatus::IOStatusT status ) const;

    protected:
        WLIOStatusInterpreter();
    };
}

#endif  // WLIOSTATUS_H_
