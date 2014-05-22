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

#ifndef WFIFFSTREAM_H_
#define WFIFFSTREAM_H_

#include <boost/shared_ptr.hpp>

#include <QtCore/qiodevice.h>

#include <fiff/fiff_proj.h>
#include <fiff/fiff_stream.h>

using namespace FIFFLIB;

/**
 * FiffStream provides an interface for reading from and writing to fiff files.
 */
class WFiffStream: public FiffStream
{
public:

    /**
     * A shared pointer on a WFiffStream.
     */
    typedef boost::shared_ptr< WFiffStream > SPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a fiff stream that uses the I/O device p_pIODevice.
     *
     * @param[in] p_pIODevice    A fiff IO device like a fiff QFile or QTCPSocket
     */
    WFiffStream( QIODevice *p_pIODevice );

    /**
     * Read the measurement info
     * Source is assumed to be an open fiff file.
     *
     * @param[in] p_Node       The node of interest
     * @param[out] p_Info      The read measurement info
     * @param[out] p_NodeInfo  The to measurement corresponding fiff_dir_tree.
     *
     * @return the to measurement corresponding fiff_dir_tree.
     */
    bool read_meas_info( const FiffDirTree& p_Node, FiffInfo& p_Info, FiffDirTree& p_NodeInfo );

    /**
     * fiff_read_proj
     *
     * [ projdata ] = fiff_read_proj(fid,node)
     *
     * Read the SSP data under a given directory node
     *
     * @param[in] p_Node    The node of interest
     *
     * @return a list of SSP projectors
     */
    QList< FiffProj > read_proj( const FiffDirTree& p_Node );

};

#endif /* WFIFFSTREAM_H_ */
