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

#ifndef WLEFTCOMMAND_H_
#define WLEFTCOMMAND_H_

#include <ostream>
#include <set>
#include <string>

namespace WLEFTCommand
{
    /**
     * An enum describing the FieldTrip message command types.
     */
    enum Enum
    {
        /**
         * Unspecified command type.
         */
        UNSPECIFIED = 0, //!< UNSPECIFIED

        /**
         * Put header.
         */
        PUTHDR = 257,  //!< PUT_HDR

        /**
         * Put data.
         */
        PUTDAT = 258,  //!< PUT_DAT

        /**
         * Put events.
         */
        PUTEVT = 259,  //!< PUT_EVT

        /**
         * Put successful.
         */
        PUTOK = 260,   //!< PUT_OK

        /**
         * Put failed.
         */
        PUTERR = 261,  //!< PUT_ERR

        /**
         * Get header.
         */
        GETHDR = 513,  //!< GET_HDR

        /**
         * Get data.
         */
        GETDAT = 514,  //!< GET_DAT

        /**
         * Get events.
         */
        GETEVT = 515,  //!< GET_EVT

        /**
         * Get successful.
         */
        GETOK = 516,   //!< GET_OK

        /**
         * Get failed.
         */
        GETERR = 517,  //!< GET_ERR

        /**
         * Flush all (header + data + events).
         */
        FLUSHHDR = 769,  //!< FLUSH_HDR

        /**
         * Flush data.
         */
        FLUSHDAT = 770,  //!< FLUSH_DAT

        /**
         * Flush events.
         */
        FLUSHEVT = 771,  //!< FLUSH_EVT

        /**
         * Flush successful.
         */
        FLUSHOK = 772, //!< FLUSH_OK

        /**
         * Flush failed.
         */
        FLUSHERR = 773, //!< FLUSH_ERR

        /**
         * Wait for data & events.
         */
        WAITDAT = 1026, //!< WAIT_DAT

        /**
         * Wait successful.
         */
        WAITOK = 1028, //!< WAIT_OK

        /**
         * Wait failed.
         */
        WAITERR = 1029 //!< WAIT_ERR
    };

    /**
     * A container of WLEFTCommand::Enum values.
     */
    typedef std::set< Enum > ContainerT;

    /**
     * Returns a container with all possible value.
     *
     * @return A value container.
     */
    ContainerT values();

    /**
     * Gets the appropriate name of the value.
     *
     * @param val The WLEFTCommand::Enum value.
     * @return The name
     */
    std::string name( Enum val );

    /**
     * Overrides the concatenation operator for console outputs.
     *
     * @param strm
     * @param obj
     * @return
     */
    std::ostream& operator<<( std::ostream &strm, const WLEFTCommand::Enum& obj );

} /* namespace WLEFTCommand */

inline std::ostream& WLEFTCommand::operator<<( std::ostream &strm, const WLEFTCommand::Enum& obj )
{
    strm << WLEFTCommand::name( obj );
    return strm;
}

#endif /* WLEFTCOMMAND_H_ */
