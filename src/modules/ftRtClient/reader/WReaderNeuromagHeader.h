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

#ifndef WREADERNEUROMAGHEADER_H_
#define WREADERNEUROMAGHEADER_H_

#include <boost/shared_ptr.hpp>

#include <QtCore/qbytearray.h>
#include <QtCore/qbuffer.h>
#include <QtCore/qfile.h>

#include <fiff/fiff_info.h>
#include <fiff/fiff_stream.h>
#include <fiff/fiff_tag.h>

#include "WFiffStream.h"

using namespace FIFFLIB;

/**
 * WReaderNeuromagHeader is a reader for binary FIFF files. It extracts he measurement information tag from the file.
 * The reader supports little endian byte order files only.
 */
class WReaderNeuromagHeader
{
public:

    /**
     * A shared pointer on a WReaderNeuromagHeader.
     */
    typedef boost::shared_ptr< WReaderNeuromagHeader > SPtr;

    /**
     * A shared pointer on a constant WReaderNeuromagHeader.
     */
    typedef boost::shared_ptr< const WReaderNeuromagHeader > ConstSPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WReaderNeuromagHeader.
     *
     * @param fname The file name.
     */
    explicit WReaderNeuromagHeader( std::string fname );

    /**
     * Constructs a new WReaderNeuromagHeader.
     *
     * @param data The data pointer.
     * @param size The size of the memory storage.
     */
    explicit WReaderNeuromagHeader( const char* data, size_t size );

    /**
     * Destroys the WReaderNeuromagHeader.
     */
    virtual ~WReaderNeuromagHeader();

    /**
     * Reads a FIFF file and extracts the measurement information.
     *
     * Inherited method from WReader.
     *
     * @param out The measurement information.
     * @return Return true if the file could be read, else false.
     */
    bool read( FIFFLIB::FiffInfo* const out );

protected:

    /**
     * Create the directory tree structure
     *
     * @param[in] p_pStream the opened fiff file
     * @param[in] p_Dir the dir entries of which the tree should be constructed
     * @param[out] p_Tree the created dir tree
     * @param[in] start dir entry to start (optional, by default 0)
     *
     * @return index of the last read dir entry
     */
    qint32 make_dir_tree( QList< FiffDirEntry >& p_Dir, FiffDirTree& p_Tree, qint32 start = 0 );

    /**
     * The WFiffStream to read from the Neuromag Header Fiff-file. Depending on the constructor call the stream can be placed on a QFile of a QBuffer.
     */
    WFiffStream::SPtr m_stream;

};

#endif /* WREADERNEUROMAGHEADER_H_ */
