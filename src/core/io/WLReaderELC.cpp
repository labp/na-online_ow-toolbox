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
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/exceptions/WTypeMismatch.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/common/WLogger.h>
#include <core/common/WStringUtils.h>

#include "core/util/WLGeometry.h"

#include "WLReaderELC.h"

using std::ifstream;
using std::string;
using std::vector;

const std::string WLReaderELC::CLASS = "WLReaderELC";

WLReaderELC::WLReaderELC( std::string fname ) :
                WReader( fname )
{
}

WLIOStatus::IOStatusT WLReaderELC::read( WLPositions* const posOut, std::vector< std::string >* const labelsOut,
                std::vector< WVector3i >* const facesOut )
{
    ifstream ifs;
    ifs.open( m_fname.c_str(), ifstream::in );

    if( !ifs || ifs.bad() )
    {
        return WLIOStatus::ERROR_FOPEN;
    }

    string line;
    size_t countPos = 0, countPoly = 0;
    bool hasPos = false, hasLabels = false, hasFaces = false;
    WLIOStatus::IOStatusT rc = WLIOStatus::ERROR_UNKNOWN;
    WLEExponent::Enum exp = WLEExponent::BASE;
    try
    {
        while( getline( ifs, line ) && ( !hasPos || !hasLabels || !hasFaces ) )
        {
            if( line.find( "UnitPosition" ) == 0 )
            {
                rc = readUnit( &exp, line );
                posOut->exponent( exp );
                posOut->unit( WLEUnit::METER );
            }
            else
                if( line.substr( 0, 16 ) == "NumberPositions=" )
                {
                    rc = readNumPos( &countPos, line );
                }
                else
                    if( line.substr( 0, 9 ) == "Positions" )
                    {
                        rc = readPositions( ifs, countPos, posOut );
                        hasPos = true;
                    }
                    else
                        if( line.substr( 0, 6 ) == "Labels" )
                        {
                            rc = readLabels( ifs, countPos, labelsOut );
                            hasLabels = true;
                        }
                        else
                            if( line.substr( 0, 15 ) == "NumberPolygons=" )
                            {
                                rc = readNumPoly( &countPoly, line );
                            }
                            else
                                if( line.substr( 0, 8 ) == "Polygons" )
                                {
                                    rc = readPolygons( ifs, countPoly, facesOut );
                                }
        }
    }
    catch( const WTypeMismatch& e )
    {
        wlog::error( CLASS ) << e.what();
        rc = WLIOStatus::ERROR_UNKNOWN;
    }

    if( facesOut->empty() )
    {
        wlog::warn( CLASS ) << "No faces found! Faces will be generated.";
        WLGeometry::computeTriangulation( facesOut, posOut->data() );
    }

    ifs.close();
    return rc;
}

WLIOStatus::IOStatusT WLReaderELC::readUnit( WLEExponent::Enum* const exp, const string& line )
{
    vector< string > tokens = string_utils::tokenize( line );
    string unit = tokens.at( 1 );
    wlog::debug( CLASS ) << "Unit: " << unit;
    if( unit.find( "mm" ) != string::npos )
    {
        *exp = WLEExponent::MILLI;
        return WLIOStatus::SUCCESS;
    }
    else
        if( unit.find( "m" ) != string::npos )
        {
            *exp = WLEExponent::BASE;
            return WLIOStatus::SUCCESS;
        }
        else
        {
            wlog::warn( CLASS ) << "Unknown unit.";
            return WLIOStatus::ERROR_UNKNOWN;
        }

}

WLIOStatus::IOStatusT WLReaderELC::readNumPos( size_t* const count, const string& line )
{
    vector< string > tokens = string_utils::tokenize( line );
    *count = string_utils::fromString< size_t >( tokens.at( 1 ) );
    wlog::debug( CLASS ) << "Number of positions: " << *count;
    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderELC::readNumPoly( size_t* const count, const string& line )
{
    vector< string > tokens = string_utils::tokenize( line );
    *count = string_utils::fromString< size_t >( tokens.at( 1 ) );
    wlog::debug( CLASS ) << "Number of polygons: " << *count;
    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderELC::readPositions( ifstream& ifs, size_t count, WLPositions* const posOut )
{
    // Check output data
    if( !posOut )
    {
        wlog::error( CLASS ) << "Position vector is null!";
        return WLIOStatus::ERROR_UNKNOWN;
    }
    posOut->resize( count );

    string line;
    for( size_t i = 0; i < count && getline( ifs, line ); ++i )
    {
        vector< string > lineTokens = string_utils::tokenize( line, ":" );

        vector< string > posTokens = string_utils::tokenize( lineTokens.back() );
        float posX = string_utils::fromString< float >( posTokens.at( posTokens.size() - 3 ) );
        float posY = string_utils::fromString< float >( posTokens.at( posTokens.size() - 2 ) );
        float posZ = string_utils::fromString< float >( posTokens.at( posTokens.size() - 1 ) );
        posOut->data().col( i ).x() = posX;
        posOut->data().col( i ).y() = posY;
        posOut->data().col( i ).z() = posZ;
    }
    if( posOut->size() < count )
    {
        return WLIOStatus::ERROR_FREAD;
    }

    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderELC::readLabels( ifstream& ifs, size_t count, std::vector< std::string >* const labelsOut )
{
    // Check output data
    if( !labelsOut )
    {
        wlog::error( CLASS ) << "Label vector is null!";
        return WLIOStatus::ERROR_UNKNOWN;
    }
    labelsOut->reserve( count );

    string line;
    for( size_t i = 0; i < count && getline( ifs, line ); ++i )
    {
        vector< std::string > labelTokens = string_utils::tokenize( line );
        for( size_t j = 0; j < labelTokens.size(); ++j )
        {
            ++i;
            labelsOut->push_back( labelTokens[j] );
        }
    }
    if( labelsOut->size() < count )
    {
        return WLIOStatus::ERROR_FREAD;
    }

    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderELC::readPolygons( ifstream& ifs, size_t count, std::vector< WVector3i >* const facesOut )
{
    // Check output data
    if( !facesOut )
    {
        wlog::error( CLASS ) << "Polygon vector is null!";
        return WLIOStatus::ERROR_UNKNOWN;
    }
    facesOut->reserve( count );

    string line;
    for( size_t i = 0; i < count && getline( ifs, line ); ++i )
    {
        vector< string > lineTokens = string_utils::tokenize( line, ":" );

        vector< string > indexTokens = string_utils::tokenize( lineTokens.back() );
        int indexX = string_utils::fromString< int >( indexTokens.at( indexTokens.size() - 3 ) );
        int indexY = string_utils::fromString< int >( indexTokens.at( indexTokens.size() - 2 ) );
        int indexZ = string_utils::fromString< int >( indexTokens.at( indexTokens.size() - 1 ) );
        facesOut->push_back( WVector3i( indexX, indexY, indexZ ) );
    }

    if( facesOut->size() < count )
    {
        return WLIOStatus::ERROR_FREAD;
    }

    return WLIOStatus::SUCCESS;
}
