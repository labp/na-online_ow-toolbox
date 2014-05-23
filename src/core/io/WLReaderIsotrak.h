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

#ifndef WREADERNEUROMAGISOTRAK_H_
#define WREADERNEUROMAGISOTRAK_H_

#include <boost/shared_ptr.hpp>

#include <fiff/fiff_dig_point.h>
#include <fiff/fiff_dir_tree.h>
#include <fiff/fiff_stream.h>

#include "core/container/WLList.h"
#include "core/data/WLDigPoint.h"
#include "core/io/WLReader.h"

using namespace FIFFLIB;

/**
 * The WReaderNeuromagIsotrak read a Neuromag Isotrak file in the big endian byte order and extracts the digitalization points.
 *
 * WReaderNeuromagIsotrak supports big endian files only.
 */
class WLReaderIsotrak
{
public:

    /**
     * A shared pointer on a WReaderNeuromagIsotrak.
     */
    typedef boost::shared_ptr< WLReaderIsotrak > SPtr;

    /**
     * A shared pointer on a constant WReaderNeuromagIsotrak.
     */
    typedef boost::shared_ptr< const WLReaderIsotrak > ConstSPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WReaderNeuromagIsotrak.
     *
     * @param fname The file name.
     */
    explicit WLReaderIsotrak( std::string fname );

    /**
     * Constructs a new WReaderNeuromagIsotrak.
     *
     * @param data The pointer to the memory storage.
     * @param size The size of the file.
     */
    explicit WLReaderIsotrak( const char* data, size_t size );

    /**
     * Destroys the WReaderNeuromagIsotrak.
     */
    virtual ~WLReaderIsotrak();

    /**
     * Reads the big endian Neuromag Isotrak file and fills the digitalization points.
     *
     * @param digPoints The list to fill.
     * @return Returns true if the file was read successfully, oherwise false.
     */
    WLReader::ReturnCode::Enum read( WLList< WLDigPoint >::SPtr digPoints );

protected:

    /**
     * Reads the digitalization points from the created FIFF directory tree.
     *
     * @param p_Node The FIFF directory tree.
     * @param out The digitalization points list.
     * @return Returns true if the points were found, otherwise false.
     */
    bool readDigPoints( const FiffDirTree& p_Node, WLList< WLDigPoint >::SPtr out );

    /**
     * Method to create a concrete DigPoint object.
     *
     * @param fiffDigPoint The MNE Dig Point.
     * @return Returns a concrete WLDigPoint.
     */
    WLDigPoint createDigPoint( const FiffDigPoint& fiffDigPoint );

private:

    /**
     * The FiffStream to read from the Isotrak Fiff-file. Depending on the constructor call the stream can be placed on a QFile of a QBuffer.
     */
    boost::shared_ptr< FiffStream > m_stream;
};

#endif /* WREADERNEUROMAGISOTRAK_H_ */