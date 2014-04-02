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
        PUT_HDR = 257,  //!< PUT_HDR

        /**
         * Put data.
         */
        PUT_DAT = 258,  //!< PUT_DAT

        /**
         * Put events.
         */
        PUT_EVT = 259,  //!< PUT_EVT

        /**
         * Put successful.
         */
        PUT_OK = 260,   //!< PUT_OK

        /**
         * Put failed.
         */
        PUT_ERR = 261,  //!< PUT_ERR

        /**
         * Get header.
         */
        GET_HDR = 513,  //!< GET_HDR

        /**
         * Get data.
         */
        GET_DAT = 514,  //!< GET_DAT

        /**
         * Get events.
         */
        GET_EVT = 515,  //!< GET_EVT

        /**
         * Get successful.
         */
        GET_OK = 516,   //!< GET_OK

        /**
         * Get failed.
         */
        GET_ERR = 517,  //!< GET_ERR

        /**
         * Flush all (header + data + events).
         */
        FLUSH_HDR = 769,  //!< FLUSH_HDR

        /**
         * Flush data.
         */
        FLUSH_DAT = 770,  //!< FLUSH_DAT

        /**
         * Flush events.
         */
        FLUSH_EVT = 771,  //!< FLUSH_EVT

        /**
         * Flush successful.
         */
        FLUSH_OK = 772, //!< FLUSH_OK

        /**
         * Flush failed.
         */
        FLUSH_ERR = 773, //!< FLUSH_ERR

        /**
         * Wait for data & events.
         */
        WAIT_DAT = 1026, //!< WAIT_DAT

        /**
         * Wait successful.
         */
        WAIT_OK = 1027, //!< WAIT_OK

        /**
         * Wait failed.
         */
        WAIT_ERR = 1028 //!< WAIT_ERR
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
