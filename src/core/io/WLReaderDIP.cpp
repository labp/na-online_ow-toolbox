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

#include <cstddef>
#include <string>
#include <vector>

#include <boost/smart_ptr/shared_ptr.hpp>
#include <core/common/exceptions/WTypeMismatch.h>
#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/common/WLogger.h>
#include <core/common/WStringUtils.h>

#include "core/container/WLArrayList.h"
#include "core/util/WLGeometry.h"

#include "WLReaderDIP.h"

using std::ifstream;
using std::string;
using std::vector;

const std::string WLReaderDIP::CLASS = "WLReaderDIP";

WLReaderDIP::WLReaderDIP( std::string fname ) :
                WLReaderGeneric< WLEMMSurface::SPtr >( fname )
{
}

WLIOStatus::IOStatusT WLReaderDIP::read( WLEMMSurface::SPtr* const surface )
{
    ifstream ifs;
    ifs.open( m_fname.c_str(), ifstream::in );

    if( !ifs || ifs.bad() )
    {
        return WLIOStatus::ERROR_FOPEN;
    }

    string line;
    size_t countPos = 0, countPoly = 0;
    bool hasPos = false, hasFaces = false;
    WLIOStatus::IOStatusT rc = WLIOStatus::ERROR_UNKNOWN;
    try
    {
        while( getline( ifs, line ) && ( !hasPos || !hasFaces ) )
        {
            if( line.find( "UnitPosition" ) == 0 )
            {
                rc = readUnit( *surface, line );
            }
            else
                if( line.find( "NumberPositions=" ) == 0 )
                {
                    rc = readNumPos( &countPos, line );
                }
                else
                    if( line.find( "Positions" ) == 0 )
                    {
                        rc = readPositions( ifs, countPos, *surface );
                        hasPos = true;
                    }
                    else
                        if( line.find( "NumberPolygons=" ) == 0 )
                        {
                            rc = readNumPoly( &countPoly, line );
                        }
                        else
                            if( line.find( "Polygons" ) == 0 )
                            {
                                rc = readPolygons( ifs, countPoly, *surface );
                            }
        }
    }
    catch( const WTypeMismatch& e )
    {
        wlog::error( CLASS ) << e.what();
        rc = WLIOStatus::ERROR_UNKNOWN;
    }

    if( ( *surface )->getFaces()->empty() )
    {
        wlog::warn( CLASS ) << "No faces found! Faces will be generated.";
        WLGeometry::computeTriangulation( ( *surface )->getFaces().get(), *( *surface )->getVertex() );
    }

    ifs.close();
    return rc;
}

WLIOStatus::IOStatusT WLReaderDIP::readUnit( WLEMMSurface::SPtr surface, const string& line )
{
    vector< string > tokens = string_utils::tokenize( line );
    string unit = tokens.at( 1 );
    wlog::debug( CLASS ) << "Unit: " << unit;
    if( unit.find( "mm" ) != string::npos )
    {
        surface->setVertexUnit( WLEUnit::METER );
        surface->setVertexExponent( WLEExponent::MILLI );
        return WLIOStatus::SUCCESS;
    }
    else
    {
        wlog::warn( CLASS ) << "Unknown unit.";
        return WLIOStatus::ERROR_UNKNOWN;
    }
}

WLIOStatus::IOStatusT WLReaderDIP::readNumPos( size_t* const count, const string& line )
{
    vector< string > tokens = string_utils::tokenize( line );
    *count = string_utils::fromString< size_t >( tokens.at( 1 ) );
    wlog::debug( "WReaderDIP" ) << "Number of positions: " << *count;
    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderDIP::readNumPoly( size_t* const count, const string& line )
{
    vector< string > tokens = string_utils::tokenize( line );
    *count = string_utils::fromString< size_t >( tokens.at( 1 ) );
    wlog::debug( CLASS ) << "Number of polygons: " << *count;
    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderDIP::readPositions( ifstream& ifs, size_t count, WLEMMSurface::SPtr surface )
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
        return WLIOStatus::ERROR_FREAD;
    }
    surface->setVertex( pos );

    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderDIP::readPolygons( ifstream& ifs, size_t count, WLEMMSurface::SPtr surface )
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
        return WLIOStatus::ERROR_FREAD;
    }
    surface->setFaces( faces );

    return WLIOStatus::SUCCESS;
}
