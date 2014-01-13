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

#include "WLReaderDIP.h"

#include <boost/smart_ptr/shared_ptr.hpp>
#include <core/common/exceptions/WTypeMismatch.h>
#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/common/WLogger.h>
#include <core/common/WStringUtils.h>
#include <cstddef>
#include <vector>

#include "../container/WLArrayList.h"
#include "../util/WLGeometry.h"

using namespace LaBP;
using namespace std;

const std::string WLReaderDIP::CLASS = "WLReaderDIP";

WLReaderDIP::WLReaderDIP( std::string fname ) :
                WLReader( fname )
{
}

WLReaderDIP::ReturnCode::Enum WLReaderDIP::read( WLEMMSurface::SPtr surface )
{
    ifstream ifs;
    ifs.open( m_fname.c_str(), ifstream::in );

    if( !ifs || ifs.bad() )
    {
        return ReturnCode::ERROR_FOPEN;
    }

    string line;
    size_t countPos = 0, countPoly = 0;
    bool hasPos = false, hasFaces = false;
    ReturnCode::Enum rc = ReturnCode::ERROR_UNKNOWN;
    try
    {
        while( getline( ifs, line ) && ( !hasPos || !hasFaces ) )
        {
            if( line.find( "UnitPosition" ) == 0 )
            {
                rc = readUnit( line, surface );
            }
            else
                if( line.find( "NumberPositions=" ) == 0 )
                {
                    rc = readNumPos( line, countPos );
                }
                else
                    if( line.find( "Positions" ) == 0 )
                    {
                        rc = readPositions( ifs, countPos, surface );
                        hasPos = true;
                    }
                    else
                        if( line.find( "NumberPolygons=" ) == 0 )
                        {
                            rc = readNumPoly( line, countPoly );
                        }
                        else
                            if( line.find( "Polygons" ) == 0 )
                            {
                                rc = readPolygons( ifs, countPoly, surface );
                            }
        }
    }
    catch( WTypeMismatch& e )
    {
        wlog::error( CLASS ) << e.what();
        rc = ReturnCode::ERROR_UNKNOWN;
    }

    if( surface->getFaces()->empty() )
    {
        wlog::warn( CLASS ) << "No faces found! Faces will be generated.";
        WLGeometry::computeTriangulation( surface->getFaces().get(), *surface->getVertex() );
    }

    ifs.close();
    return rc;
}

WLReaderDIP::ReturnCode::Enum WLReaderDIP::readUnit( string& line, WLEMMSurface::SPtr surface )
{
    vector< string > tokens = string_utils::tokenize( line );
    string unit = tokens.at( 1 );
    wlog::debug( CLASS ) << "Unit: " << unit;
    if( unit.find( "mm" ) != string::npos )
    {
        surface->setVertexUnit( WLEUnit::METER );
        surface->setVertexExponent( WLEExponent::MILLI );
        return ReturnCode::SUCCESS;
    }
    else
    {
        wlog::warn( CLASS ) << "Unknown unit.";
        return ReturnCode::ERROR_UNKNOWN;
    }

}

WLReaderDIP::ReturnCode::Enum WLReaderDIP::readNumPos( string& line, size_t& count )
{
    vector< string > tokens = string_utils::tokenize( line );
    count = string_utils::fromString< size_t >( tokens.at( 1 ) );
    wlog::debug( "WReaderDIP" ) << "Number of positions: " << count;
    return ReturnCode::SUCCESS;
}

WLReaderDIP::ReturnCode::Enum WLReaderDIP::readNumPoly( string& line, size_t& count )
{
    vector< string > tokens = string_utils::tokenize( line );
    count = string_utils::fromString< size_t >( tokens.at( 1 ) );
    wlog::debug( CLASS ) << "Number of polygons: " << count;
    return ReturnCode::SUCCESS;
}

WLReaderDIP::ReturnCode::Enum WLReaderDIP::readPositions( ifstream& ifs, size_t count, WLEMMSurface::SPtr surface )
{

    WLArrayList< WPosition >::SPtr pos( new WLArrayList< WPosition >() );
    pos->reserve( count );

    string line;
    for( size_t i = 0; i < count && getline( ifs, line ); ++i )
    {
        vector< string > lineTokens = string_utils::tokenize( line, ":" );

        vector< string > posTokens = string_utils::tokenize( lineTokens.back() );
        float posX = string_utils::fromString< float >( posTokens.at( posTokens.size() - 3 ) );
        float posY = string_utils::fromString< float >( posTokens.at( posTokens.size() - 2 ) );
        float posZ = string_utils::fromString< float >( posTokens.at( posTokens.size() - 1 ) );
        pos->push_back( WPosition( posX, posY, posZ ) );
    }
    if( pos->size() < count )
    {
        return ReturnCode::ERROR_FREAD;
    }
    surface->setVertex( pos );

    return ReturnCode::SUCCESS;
}

WLReaderDIP::ReturnCode::Enum WLReaderDIP::readPolygons( ifstream& ifs, size_t count, WLEMMSurface::SPtr surface )
{
    WLArrayList< WVector3i >::SPtr faces( new WLArrayList< WVector3i >() );
    faces->reserve( count );

    string line;
    for( size_t i = 0; i < count && getline( ifs, line ); ++i )
    {
        vector< string > lineTokens = string_utils::tokenize( line, ":" );

        vector< string > indexTokens = string_utils::tokenize( lineTokens.back() );
        int indexX = string_utils::fromString< int >( indexTokens.at( indexTokens.size() - 3 ) );
        int indexY = string_utils::fromString< int >( indexTokens.at( indexTokens.size() - 2 ) );
        int indexZ = string_utils::fromString< int >( indexTokens.at( indexTokens.size() - 1 ) );
        faces->push_back( WVector3i( indexX, indexY, indexZ ) );
    }
    if( faces->size() < count )
    {
        return ReturnCode::ERROR_FREAD;
    }
    surface->setFaces( faces );

    return ReturnCode::SUCCESS;
}
