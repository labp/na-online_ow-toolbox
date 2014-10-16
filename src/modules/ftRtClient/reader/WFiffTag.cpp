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

#include <QtNetwork/qtcpsocket.h>

#include "WFiffTag.h"

bool WFiffTag::read_tag_info( FiffStream* p_pStream, FiffTag::SPtr &p_pTag, bool p_bDoSkip )
{
    p_pTag = FiffTag::SPtr( new FiffTag() );

    // read tag kind
    if( 0 > p_pStream->readRawData( ( char * )&p_pTag->kind, sizeof( p_pTag->kind ) ) )
    {
        return false;
    }

    // read tag type
    if( 0 > p_pStream->readRawData( ( char * )&p_pTag->type, sizeof( p_pTag->type ) ) )
    {
        return false;
    }

    qint32 size;
    // read tags data size
    if( 0 > p_pStream->readRawData( ( char * )&size, sizeof(qint32) ) )
    {
        return false;
    }

    p_pTag->resize( size );

    // read next
    if( 0 > p_pStream->readRawData( ( char * )&p_pTag->next, sizeof( p_pTag->next ) ) )
    {
        return false;
    }

    // skip the data block
    if( p_bDoSkip )
    {
        QTcpSocket* t_qTcpSocket = qobject_cast< QTcpSocket* >( p_pStream->device() );
        if( t_qTcpSocket )
        {
            p_pStream->skipRawData( size );
        }
        else
        {
            if( p_pTag->next == FIFFV_NEXT_SEQ )
            {
                p_pStream->device()->seek( p_pStream->device()->pos() + size );
            }
            else
                if( p_pTag->next > 0 )
                {
                    p_pStream->device()->seek( p_pTag->next );
                }
        }
    }

    return true;
}

bool WFiffTag::read_tag( FiffStream* p_pStream, FiffTag::SPtr& p_pTag, qint64 pos )
{
    if( pos >= 0 )
    {
        p_pStream->device()->seek( pos );
    }

    p_pTag = FiffTag::SPtr( new FiffTag() );

    // read tag kind
    if( 0 > p_pStream->readRawData( ( char * )&p_pTag->kind, sizeof( p_pTag->kind ) ) )
    {
        return false;
    }

    // read tag type
    if( 0 > p_pStream->readRawData( ( char * )&p_pTag->type, sizeof( p_pTag->type ) ) )
    {
        return false;
    }

    qint32 size;
    // read tags data size
    if( 0 > p_pStream->readRawData( ( char * )&size, sizeof(qint32) ) )
    {
        return false;
    }

    p_pTag->resize( size );

    // read next
    if( 0 > p_pStream->readRawData( ( char * )&p_pTag->next, sizeof( p_pTag->next ) ) )
    {
        return false;
    }

    if( p_pTag->size() > 0 )
    {
        p_pStream->readRawData( p_pTag->data(), p_pTag->size() );
    }

    if( p_pTag->next != FIFFV_NEXT_SEQ )
        p_pStream->device()->seek( p_pTag->next );

    return true;
}
