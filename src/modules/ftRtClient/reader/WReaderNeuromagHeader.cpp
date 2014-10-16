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

#include <cstdio>
#include <iostream>
#include <string>

#include <QtCore/qlist.h>

#include <fiff/fiff.h>
#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_dir_entry.h>
#include <fiff/fiff_dir_tree.h>

#include <core/common/WIOTools.h>
#include <core/common/WLogger.h>
#include <core/dataHandler/exceptions/WDHNoSuchFile.h>

#include "WFiffTag.h"
#include "WFiffStream.h"
#include "WReaderNeuromagHeader.h"

using namespace std;
using namespace FIFFLIB;

const std::string WReaderNeuromagHeader::CLASS = "WReaderNeuromagHeader";

WReaderNeuromagHeader::WReaderNeuromagHeader( std::string fname )
{
    if( !fileExists( fname ) )
    {
        throw WDHNoSuchFile( fname );
    }

    m_stream.reset( new WFiffStream( new QFile( QString::fromStdString( fname ) ) ) );
}

WReaderNeuromagHeader::WReaderNeuromagHeader( const char* data, size_t size )
{
    m_stream.reset( new WFiffStream( new QBuffer( new QByteArray( data, size ) ) ) );
}

WReaderNeuromagHeader::~WReaderNeuromagHeader()
{
}

bool WReaderNeuromagHeader::read( FIFFLIB::FiffInfo* const out )
{
    FiffDirTree tree;
    QList< FiffDirEntry > tags;

    m_stream->setByteOrder( QDataStream::LittleEndian ); // set the byte order.

    wlog::debug( CLASS ) << "Buffer size: " << m_stream->device()->size();
    wlog::debug( CLASS ) << "Byte Order [0 = Big, 1 = Little]: " << m_stream->byteOrder();

    if( m_stream->device()->open( QIODevice::ReadOnly ) )
    {
        wlog::debug( CLASS ) << "Stream opened.";
    }
    else
    {
        wlog::debug( CLASS ) << "Stream not opened.";
        return false;
    }

    m_stream->device()->seek( 0 );

    WFiffTag tag;

    // read file_id tag.
    WFiffTag::read_tag_info( &tag, m_stream.get() );

    if( tag.kind != FIFF_FILE_ID )
    {
        wlog::error( CLASS ) << "File does not start with a file id tag";
        return false;
    }

    if( tag.type != FIFFT_ID_STRUCT )
    {
        wlog::error( CLASS ) << "File does not start with a file id tag";
        return false;
    }
    if( tag.size() != 20 )
    {
        wlog::error( CLASS ) << "File does not start with a file id tag";
        return false;
    }

    // read dir_pointer tag.
    WFiffTag::read_tag( &tag, m_stream.get() );
    if( tag.kind != FIFF_DIR_POINTER )
    {
        wlog::error( CLASS ) << "File does have a directory pointer";
        return false;
    }

    tags.clear();
    qint32 dirpos = *tag.toInt();
    if( dirpos > 0 )
    {
        WFiffTag::read_tag( &tag, m_stream.get(), dirpos );
        tags = tag.toDirEntry();
    }
    else
    {
        m_stream->device()->seek( 0 );
        FiffDirEntry t_fiffDirEntry;
        while( tag.next >= 0 )
        {
            t_fiffDirEntry.pos = m_stream->device()->pos();
            WFiffTag::read_tag_info( &tag, m_stream.get() );

            t_fiffDirEntry.kind = tag.kind;
            t_fiffDirEntry.type = tag.type;
            t_fiffDirEntry.size = tag.size();
            tags.append( t_fiffDirEntry );
        }
    }

    make_dir_tree( &tree, tags ); // build directory structure

    FiffDirTree nodeInfo;

    wlog::debug( CLASS ) << "Measurement information read.";

    m_stream->read_meas_info( tree, *out, nodeInfo );

    // clean up
    m_stream->device()->seek( 0 );
    m_stream->device()->close();

    return true;
}

qint32 WReaderNeuromagHeader::make_dir_tree( FIFFLIB::FiffDirTree* const p_Tree, const QList< FIFFLIB::FiffDirEntry >& p_Dir,
                qint32 start )
{
    p_Tree->clear();

    WFiffTag t_pTag;

    qint32 block;
    if( p_Dir[start].kind == FIFF_BLOCK_START )
    {
        WFiffTag::read_tag( &t_pTag, m_stream.get(), p_Dir[start].pos );
        block = *t_pTag.toInt();
    }
    else
    {
        block = 0;
    }

    qint32 current = start;

    p_Tree->block = block;
    p_Tree->nent = 0;
    p_Tree->nchild = 0;

    while( current < p_Dir.size() )
    {
        if( p_Dir[current].kind == FIFF_BLOCK_START )
        {
            if( current != start )
            {
                FiffDirTree t_ChildTree;
                current = make_dir_tree( &t_ChildTree, p_Dir, current );
                ++( *p_Tree ).nchild;
                p_Tree->children.append( t_ChildTree );
            }
        }
        else
            if( p_Dir[current].kind == FIFF_BLOCK_END )
            {
                WFiffTag::read_tag( &t_pTag, m_stream.get(), p_Dir[start].pos );
                if( *t_pTag.toInt() == p_Tree->block )
                    break;
            }
            else
            {
                ++( *p_Tree ).nent;
                p_Tree->dir.append( p_Dir[current] );

                //
                //  Add the id information if available
                //
                if( block == 0 )
                {
                    if( p_Dir[current].kind == FIFF_FILE_ID )
                    {
                        WFiffTag::read_tag( &t_pTag, m_stream.get(), p_Dir[current].pos );
                        p_Tree->id = t_pTag.toFiffID();
                    }
                }
                else
                {
                    if( p_Dir[current].kind == FIFF_BLOCK_ID )
                    {
                        WFiffTag::read_tag( &t_pTag, m_stream.get(), p_Dir[current].pos );
                        p_Tree->id = t_pTag.toFiffID();
                    }
                    else
                        if( p_Dir[current].kind == FIFF_PARENT_BLOCK_ID )
                        {
                            WFiffTag::read_tag( &t_pTag, m_stream.get(), p_Dir[current].pos );
                            p_Tree->parent_id = t_pTag.toFiffID();
                        }
                }
            }
        ++current;
    }

    //
    // Eliminate the empty directory
    //
    if( p_Tree->nent == 0 )
        p_Tree->dir.clear();

    return current;
}
