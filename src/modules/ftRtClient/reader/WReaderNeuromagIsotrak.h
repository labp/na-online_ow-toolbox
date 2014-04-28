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

#include <core/dataHandler/io/WReader.h>

#include "core/container/WLList.h"
#include "core/data/WLDigPoint.h"

using namespace FIFFLIB;

/**
 * The WReaderNeuromagIsotrak read a Neuromag Isotrak file in the big endian byte order and extracts the digitalization points.
 *
 * WReaderNeuromagIsotrak supports big endian files only.
 */
class WReaderNeuromagIsotrak: public WReader
{
public:

    /**
     * A shared pointer on a WReaderNeuromagIsotrak.
     */
    typedef boost::shared_ptr< WReaderNeuromagIsotrak > SPtr;

    /**
     * A shared pointer on a constant WReaderNeuromagIsotrak.
     */
    typedef boost::shared_ptr< const WReaderNeuromagIsotrak > ConstSPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WReaderNeuromagIsotrak.
     *
     * @param fname The file name.
     */
    explicit WReaderNeuromagIsotrak( std::string fname );

    /**
     * Destroys the WReaderNeuromagIsotrak.
     */
    virtual ~WReaderNeuromagIsotrak();

    /**
     * Reads the big endian Neuromag Isotrak file and fills the digitalization points.
     *
     * @param digPoints The list to fill.
     * @return Returns true if the file was read successfully, oherwise false.
     */
    bool read( WLList< WLDigPoint >::SPtr& digPoints );

protected:

    /**
     * Reads the digitalization points from the created FIFF directory tree.
     *
     * @param p_Node The FIFF directory tree.
     * @param out The digitalization points list.
     * @return Returns true if the points were found, otherwise false.
     */
    bool readDigPoints( FiffStream& stream, const FiffDirTree& p_Node, WLList< WLDigPoint >& out );

    WLDigPoint createDigPoint( const FiffDigPoint& fiffDigPoint );
};

#endif /* WREADERNEUROMAGISOTRAK_H_ */
