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

#include <cstdio>
#include <iostream>

#include <QtCore/qbytearray.h>
#include <QtCore/qfile.h>
#include <QtCore/qlist.h>

#include <fiff/fiff.h>
#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_dir_entry.h>
#include <fiff/fiff_dir_tree.h>

#include "core/common/WLogger.h"

#include "WFiffTag.h"
#include "WFiffStream.h"
#include "WReaderNeuromagHeader.h"

using namespace std;
using namespace FIFFLIB;

const std::string WReaderNeuromagHeader::CLASS = "WReaderNeuromagHeader";

WReaderNeuromagHeader::WReaderNeuromagHeader( std::string fname ) :
                WReader( fname )
{
}

WReaderNeuromagHeader::~WReaderNeuromagHeader()
{
}

bool WReaderNeuromagHeader::read( FIFFLIB::FiffInfo* const out )
{
    wlog::debug( CLASS ) << "Start Up.";

    QFile file( QString::fromStdString( m_fname ) );

    WFiffStream stream( &file ); // create a stream on the file.
    FiffDirTree tree;
    QList< FiffDirEntry > tags;

    stream.setByteOrder( QDataStream::LittleEndian ); // set the byte order.

    wlog::debug( CLASS ) << "File name: " << m_fname;
    wlog::debug( CLASS ) << "File size: " << file.size() << " Byte";
    wlog::debug( CLASS ) << "Byte Order [0 = Big, 1 = Little]: " << stream.byteOrder();

    wlog::debug( CLASS ) << "Begin reading.";

    if( stream.device()->open( QIODevice::ReadOnly ) )
        wlog::debug( CLASS ) << "Stream opened.";
    else
    {
        wlog::debug( CLASS ) << "Stream not opened.";
        return false;
    }

    WFiffTag::SPtr tag = FiffTag::SPtr( new FiffTag() );

    // read file_id tag.
    WFiffTag::read_tag_info( &stream, tag );

    if( tag->kind != FIFF_FILE_ID )
    {
        wlog::error( CLASS ) << "File does not start with a file id tag";
        return false;
    }

    if( tag->type != FIFFT_ID_STRUCT )
    {
        wlog::error( CLASS ) << "File does not start with a file id tag";
        return false;
    }
    if( tag->size() != 20 )
    {
        wlog::error( CLASS ) << "File does not start with a file id tag";
        return false;
    }

    // read dir_pointer tag.
    WFiffTag::read_tag( &stream, tag );
    if( tag->kind != FIFF_DIR_POINTER )
    {
        wlog::error( CLASS ) << "File does have a directory pointer";
        return false;
    }

    wlog::debug( CLASS ) << "Create directory tree.";

    tags.clear();
    qint32 dirpos = *tag->toInt();
    if( dirpos > 0 )
    {
        WFiffTag::read_tag( &stream, tag, dirpos );
        tags = tag->toDirEntry();
    }
    else
    {
        stream.device()->seek( 0 );
        FiffDirEntry t_fiffDirEntry;
        while( tag->next >= 0 )
        {
            t_fiffDirEntry.pos = stream.device()->pos();
            WFiffTag::read_tag_info( &stream, tag );

            t_fiffDirEntry.kind = tag->kind;
            t_fiffDirEntry.type = tag->type;
            t_fiffDirEntry.size = tag->size();
            tags.append( t_fiffDirEntry );
        }
    }

    make_dir_tree( &stream, tags, tree ); // build directory structure

    FiffDirTree nodeInfo;

    wlog::debug( CLASS ) << "Read measurement information.";

    stream.read_meas_info( tree, *out, nodeInfo );

    wlog::debug( CLASS ) << "Finished reading.";

    // clean up
    stream.device()->seek( 0 );
    stream.device()->close();
    wlog::debug( CLASS ) << "Stream closed.";

    return true;
}

qint32 WReaderNeuromagHeader::make_dir_tree( FiffStream* p_pStream, QList< FiffDirEntry >& p_Dir, FiffDirTree& p_Tree,
                qint32 start )
{
    p_Tree.clear();

    WFiffTag::SPtr t_pTag;

    qint32 block;
    if( p_Dir[start].kind == FIFF_BLOCK_START )
    {
        WFiffTag::read_tag( p_pStream, t_pTag, p_Dir[start].pos );
        block = *t_pTag->toInt();
    }
    else
    {
        block = 0;
    }

    qint32 current = start;

    p_Tree.block = block;
    p_Tree.nent = 0;
    p_Tree.nchild = 0;

    while( current < p_Dir.size() )
    {
        if( p_Dir[current].kind == FIFF_BLOCK_START )
        {
            if( current != start )
            {
                FiffDirTree t_ChildTree;
                current = make_dir_tree( p_pStream, p_Dir, t_ChildTree, current );
                ++p_Tree.nchild;
                p_Tree.children.append( t_ChildTree );
            }
        }
        else
            if( p_Dir[current].kind == FIFF_BLOCK_END )
            {
                WFiffTag::read_tag( p_pStream, t_pTag, p_Dir[start].pos );
                if( *t_pTag->toInt() == p_Tree.block )
                    break;
            }
            else
            {
                ++p_Tree.nent;
                p_Tree.dir.append( p_Dir[current] );

                //
                //  Add the id information if available
                //
                if( block == 0 )
                {
                    if( p_Dir[current].kind == FIFF_FILE_ID )
                    {
                        WFiffTag::read_tag( p_pStream, t_pTag, p_Dir[current].pos );
                        p_Tree.id = t_pTag->toFiffID();
                    }
                }
                else
                {
                    if( p_Dir[current].kind == FIFF_BLOCK_ID )
                    {
                        WFiffTag::read_tag( p_pStream, t_pTag, p_Dir[current].pos );
                        p_Tree.id = t_pTag->toFiffID();
                    }
                    else
                        if( p_Dir[current].kind == FIFF_PARENT_BLOCK_ID )
                        {
                            WFiffTag::read_tag( p_pStream, t_pTag, p_Dir[current].pos );
                            p_Tree.parent_id = t_pTag->toFiffID();
                        }
                }
            }
        ++current;
    }

    //
    // Eliminate the empty directory
    //
    if( p_Tree.nent == 0 )
        p_Tree.dir.clear();

    return current;
}
