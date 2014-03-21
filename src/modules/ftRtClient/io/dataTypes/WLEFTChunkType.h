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

#ifndef WLEFTCHUNKTYPE_H_
#define WLEFTCHUNKTYPE_H_

#include <ostream>
#include <set>
#include <string>

namespace WLEFTChunkType
{

    /**
     * An enum describing the FieldTrip chunk types.
     */
    enum Enum
    {
        /**
         * This chunk can represent unspecified binary data, which can for example be used during development of site-specific protocols.
         * The buffer server will make no attempt to interpret the contents.
         */
        FT_CHUNK_UNSPECIFIED = 0,       //!< FT_CHUNK_UNSPECIFIED

        /**
         * This chunk is used for labeling the channels (or elements of the feature vector) that are represented in the buffer.
         */
        FT_CHUNK_CHANNEL_NAMES = 1,     //!< FT_CHUNK_CHANNEL_NAMES

        /**
         * This chunk is useful for specifying that a channel can have one of a discrete number of different types.
         */
        FT_CHUNK_CHANNEL_FLAGS = 2,     //!< FT_CHUNK_CHANNEL_FLAGS

        /**
         * This chunk describes the mapping from A/D values to physical quantities such as micro-Volts in EEG.
         */
        FT_CHUNK_RESOLUTIONS = 3,       //!< FT_CHUNK_RESOLUTIONS

        /**
         * This contains an arbitrary number of key/value pairs, each of which is given as a 0-terminated string.
         * An empty key (=double 0) indicates the end of the list.
         */
        FT_CHUNK_ASCII_KEYVAL = 4,      //!< FT_CHUNK_ASCII_KEYVAL

        /**
         * Used for transporting a NIFTI-1 header structure for specifying fMRI data.
         */
        FT_CHUNK_NIFTI1 = 5,            //!< FT_CHUNK_NIFTI1

        /**
         * Used for transporting the sequence protocol used in Siemens MR scanners (VB17).
         * This is also part of the DICOM header (private tag 0029:0120) that these scanners write.
         */
        FT_CHUNK_SIEMENS_AP = 6,        //!< FT_CHUNK_SIEMENS_AP

        /**
         * This chunk contains a .res4 file as written by the CTF MEG acquisition software in its normal binary (big-endian) format.
         */
        FT_CHUNK_CTF_RES4 = 7,          //!< FT_CHUNK_CTF_RES4

        /**
         * These chunks contain .fif files as written by the neuromag2ft real-time interface.
         * The header file is in its native platform (little-endian) format.
         */
        FT_CHUNK_NEUROMAG_HEADER = 8,   //!< FT_CHUNK_NEUROMAG_HEADER

        /**
         * These chunks contain .fif files as written by the neuromag2ft real-time interface.
         * The file is in the big-endian format.
         */
        FT_CHUNK_NEUROMAG_ISOTRAK = 9,  //!< FT_CHUNK_NEUROMAG_ISOTRAK

        /**
         * These chunks contain .fif files as written by the neuromag2ft real-time interface.
         * The file is in the big-endian format.
         */
        FT_CHUNK_NEUROMAG_HPIRESULT = 10  //!< FT_CHUNK_NEUROMAG_HPIRESULT
    };

    /**
     * A container of WLEFTChunkType::Enum values.
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
     * @param val The WLEFTChunkType::Enum value.
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
    std::ostream& operator<<( std::ostream &strm, const WLEFTChunkType::Enum& obj );

} /* namespace WLEFTChunkType */

inline std::ostream& WLEFTChunkType::operator<<( std::ostream &strm, const WLEFTChunkType::Enum& obj )
{
    strm << WLEFTChunkType::name( obj );
    return strm;
}

#endif /* WLEFTCHUNKTYPE_H_ */
