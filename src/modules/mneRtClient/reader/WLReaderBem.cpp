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

#include <boost/shared_ptr.hpp>
#include <QtCore/QFile>
#include <QtCore/QList>
#include <QtCore/QString>

#include <mne/mne_surface.h>

#include <core/common/WLogger.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLReaderBem.h"

using namespace MNELIB;
using namespace LaBP;

const std::string WLReaderBem::CLASS = "WLReaderBem";

WLReaderBem::WLReaderBem( std::string fname ) throw( WDHNoSuchFile ) :
                WReader( fname )
{
}

WLReaderBem::~WLReaderBem()
{
}

bool WLReaderBem::read( std::vector< LaBP::WLEMMBemBoundary::SPtr >* const bems )
{
    QFile file( QString::fromStdString( m_fname ) );
    QList< MNESurface::SPtr > surfaces;
    if( !MNESurface::read( file, surfaces ) )
    {
        wlog::error( CLASS ) << "Could not load BEM layers!" << endl;
        return false;
    }

    QList< MNESurface::SPtr >::ConstIterator it;
    for( it = surfaces.begin(); it != surfaces.end(); ++it )
    {
        WLEMMBemBoundary::SPtr bem( new WLEMMBemBoundary() );
        bem->setBemType( getTypeFromBemId( ( *it )->id ) );
        bem->setConductivity( ( *it )->sigma );

        boost::shared_ptr< std::vector< WPosition > > vertex( new std::vector< WPosition > );
        const MNESurface::PointsT::Index nr_ver = ( *it )->rr.cols();
        const double factor = 1000;
        vertex->reserve( nr_ver );
        for( MNESurface::PointsT::Index i = 0; i < nr_ver; ++i )
        {
            WPosition pos( ( *it )->rr( 0, i ) * factor, ( *it )->rr( 1, i ) * factor, ( *it )->rr( 2, i ) * factor );
            vertex->push_back( pos );
        }
        bem->setVertex( vertex );
        bem->setVertexExponent( WEExponent::MILLI );
        bem->setVertexUnit( WEUnit::METER );

        boost::shared_ptr< std::vector< WVector3i > > faces( new std::vector< WVector3i > );
        const MNESurface::PointsT::Index nr_tri = ( *it )->tris.cols();
        faces->reserve( nr_tri );
        for( MNESurface::TrianglesT::Index i = 0; i < nr_tri; ++i )
        {
            // TODO(pieloth): check start index 0 or 1. without -1 triangulation throws a segfault.
            WVector3i tri( ( *it )->tris( 0, i ) - 1, ( *it )->tris( 1, i ) - 1, ( *it )->tris( 2, i ) - 1 );
            faces->push_back( tri );
        }
        bem->setFaces( faces );

        bems->push_back( bem );

        wlog::debug( CLASS ) << "Adding BEM: " << WEBemType::name( bem->getBemType() ) << "; " << vertex->size() << "; "
                        << faces->size();
    }

    return true;
}

WEBemType::Enum WLReaderBem::getTypeFromBemId( int id )
{
    switch( id )
    {
        case -1:
            return WEBemType::UNKNOWN2;
        case 0:
            return WEBemType::UNKNOWN;
        case 1:
            return WEBemType::BRAIN;
            // case 2:
        case 3:
            return WEBemType::SKULL;
        case 4:
            return WEBemType::OUTER_SKIN;
        default:
            wlog::error( CLASS ) << "Unknown BEM id!";
            return WEBemType::UNKNOWN;
    }
}
