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

#include <QtCore/qfile.h>
#include <QtCore/qlist.h>
#include <QtCore/qstring.h>

#include <fiff/fiff.h>
#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_dir_entry.h>
#include <fiff/fiff_dir_tree.h>
#include <fiff/fiff_stream.h>
#include <fiff/fiff_tag.h>

#include <core/common/WLogger.h>

#include "core/data/enum/WLEPointType.h"

#include "WReaderNeuromagIsotrak.h"

using namespace std;
using namespace FIFFLIB;

const std::string WReaderNeuromagIsotrak::CLASS = "WReaderNeuromagIsotrak";

WReaderNeuromagIsotrak::WReaderNeuromagIsotrak( std::string fname ) :
                WReader( fname )
{

}

WReaderNeuromagIsotrak::~WReaderNeuromagIsotrak()
{

}

bool WReaderNeuromagIsotrak::read( WLList< WLDigPoint >::SPtr digPoints )
{
    wlog::debug( CLASS ) << "Start Up.";

    QFile file( QString::fromStdString( m_fname ) );

    digPoints.reset( new WLList< WLDigPoint > );
    FiffStream stream( &file );
    FiffDirTree tree;
    QList< FiffDirEntry > tags;

    if( stream.open( tree, tags ) )
    {
        wlog::debug( CLASS ) << "Stream opened.";
    }
    else
    {
        wlog::debug( CLASS ) << "Stream not opened.";
        return false;
    }

    return true;
}

bool WReaderNeuromagIsotrak::readDigPoints( FiffStream& stream, const FiffDirTree& p_Node, WLList< WLDigPoint >& out )
{
    out.clear();

    //
    //   Find the desired blocks
    //
    QList< FiffDirTree > isotrak = p_Node.dir_tree_find( FIFFB_ISOTRAK );

    if( isotrak.size() == 0 )
    {
        wlog::error( CLASS ) << "Could not find Isotrak data.";
        return false;
    }

    //
    //  Read Isotrak data.
    //
    FiffTag::SPtr t_pTag;
    QList< FiffChInfo > chs;
    fiff_int_t kind = -1;
    fiff_int_t pos = -1;

    for( qint32 k = 0; k < isotrak[0].nent; ++k )
    {
        kind = isotrak[0].dir[k].kind;
        pos = isotrak[0].dir[k].pos;
        switch( kind )
        {
            case FIFF_CH_INFO:
                FiffTag::read_tag( &stream, t_pTag, pos );
                chs.append( t_pTag->toChInfo() );
                break;
            case FIFF_DIG_POINT:
                FiffTag::read_tag( &stream, t_pTag, pos );
                const FiffDigPoint point = t_pTag->toDigPoint();
                out.push_back( createDigPoint( point ) );
                break;
        }
    }

    return out.size() > 0;
}

WLDigPoint WReaderNeuromagIsotrak::createDigPoint( const FiffDigPoint& fiffDigPoint )
{
    const WLDigPoint::PointT p( fiffDigPoint.r[0], fiffDigPoint.r[1], fiffDigPoint.r[2] );
    const WLDigPoint dig( p, fiffDigPoint.kind, fiffDigPoint.ident );

    return dig;
}
