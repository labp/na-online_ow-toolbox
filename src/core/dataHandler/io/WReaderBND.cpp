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



#include <cstddef>

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <osg/Array>
#include <osg/Vec3>

#include <core/graphicsEngine/WTriangleMesh.h>
#include <core/common/WLogger.h>
#include <core/common/WStringUtils.h>
#include <core/dataHandler/exceptions/WDHIOFailure.h>
#include <core/dataHandler/exceptions/WDHNoSuchFile.h>
#include <core/dataHandler/exceptions/WDHParseError.h>
#include <core/dataHandler/WEEGPositionsLibrary.h>

#include "WReaderBND.h"


WReaderBND::WReaderBND( std::string fname )
    : WReader( fname )
{
}

boost::shared_ptr< WEEGPositionsLibrary > WReaderBND::read()
{
    std::ifstream ifs;
    ifs.open( m_fname.c_str(), std::ifstream::in );
    std::cout << m_fname.c_str() << std::endl;

    if( !ifs || ifs.bad() )
    {
        throw WDHNoSuchFile( std::string( "Problem loading file " + m_fname + ". Probably file not found." ) );
    }

    std::string line;
    while( ifs.good() && line.substr( 0, 16 ) != "NumberPositions=" )  // go to number of positions
    {
        std::getline( ifs, line );
        if( !ifs.good() )
        {
            throw WDHIOFailure( std::string( "Unexpected end of file " + m_fname ) );
        }
    }

    std::vector< std::string > tokens = string_utils::tokenize( line );
    std::size_t numPositions = string_utils::fromString< std::size_t >( tokens.at( 1 ) );
    while( ifs.good() && line.substr( 0, 9 ) != "Positions" )  // go to line before start of positions
    {
        std::getline( ifs, line );
        if( !ifs.good() )
        {
            throw WDHIOFailure( std::string( "Unexpected end of file " + m_fname ) );
        }
    }

    std::vector< WPosition > positions;
    positions.reserve( numPositions );

    unsigned int counter = 0;
    while( counter != numPositions && ifs.good() && line.substr( 0, 14 ) != "NumberPolygons" )  // run through all positions
    {
        std::getline( ifs, line );
        if( !ifs.good() )
        {
            throw WDHIOFailure( std::string( "Unexpected end of file " + m_fname ) );
        }
        else
        {
        ++counter;
        std::vector< std::string > lineTokens = string_utils::tokenize( line, ":" );

        std::vector< std::string > posTokens = string_utils::tokenize( lineTokens.back() );
        double posX = string_utils::fromString< double >( posTokens.at( posTokens.size() - 3 ) );
        double posY = string_utils::fromString< double >( posTokens.at( posTokens.size() - 2 ) );
        double posZ = string_utils::fromString< double >( posTokens.at( posTokens.size() - 1 ) );
        positions.push_back( WPosition( posX, posY, posZ ) );
        }
    }

    if( positions.size() != numPositions)
    {
        throw WDHParseError( std::string( "Could not find correct number of Positions regarding "
             " to NumberPositions statement in file " + m_fname ) );
    }

    // read number of polygons statement
    std::getline( ifs, line );
    tokens = string_utils::tokenize( line );
    size_t numPolygons = string_utils::fromString< int >( tokens.at( 1 ) );

    while( ifs.good() && line.substr( 0, 8 ) != "Polygons" )  // go to line before start of polygons
    {
        std::getline( ifs, line );
        if( !ifs.good() )
        {
            throw WDHIOFailure( std::string( "Unexpected end of file " + m_fname ) );
        }
    }

    polygons.reserve( numPolygons );
    counter = 0;
    while( counter != numPolygons && ifs.good() )  // run through all polygons
    {
        std::getline( ifs, line );
        if( !ifs.good() )
        {
            throw WDHIOFailure( std::string( "Unexpected end of file " + m_fname ) );
        }
        else
        {
        ++counter;
        try
        {
        std::vector< std::string > lineTokens = string_utils::tokenize( line );
        if( lineTokens.size() != 3 )
        {
            throw WDHParseError();
        }

        int indexX = string_utils::fromString< int >( lineTokens.at( 0 ) );
        int indexY = string_utils::fromString< int >( lineTokens.at( 1 ) );
        int indexZ = string_utils::fromString< int >( lineTokens.at( 2 ) );

        std::vector<int> vec (3);
        vec[0] = indexX;
        vec[1] = indexY;
        vec[2] = indexZ;
        polygons.push_back( vec );

        }
        catch( ... )
        {
        throw WDHParseError( std::string( "Input file " + m_fname + " incorrect. E.g., check for empty lines." ) );
        }
        }
    }

    if( polygons.size()  != numPolygons)
    {
        throw WDHParseError( std::string( "Could not find correct number of Polygons regarding "
             " to NumberPolygons statement in file " + m_fname ) );
    }

    ifs.close();
    unsigned int i;
    return boost::shared_ptr< WEEGPositionsLibrary >(/* new WEEGPositionsLibrary( positions, polygons )*/ ); // TODO(pieloth)

//    // Positionsdaten in Vec3Array packen
//    osg::ref_ptr< osg::Vec3Array > positionsVec = new osg::Vec3Array;
//    positionsVec->reserve( numPositions );
//    for( i = 0; i < numPositions * 3; i += 3)
//    {
//    positionsVec->push_back( osg::Vec3( positions.at( i ),
//                                    positions.at( i + 1 ),
//                                    positions.at( i + 2 ) ) );
//    }

//    // testweise erste und letzte 10 Positions-Elemente ausgeben
//    std::cout << "First positions: ";
//    for( i = 0; i < 10; ++i )
//    {
//    std::cout << positions.at( i ) << " ";
//    }
//    std::cout << std::endl << "Last positions: ";
//    for( i = positions.size() - 10; i < positions.size(); ++i )
//    {
//    std::cout << positions.at( i ) << " ";
//    }
//    std::cout << std::endl;


//    // testweise erste und letzte 10 Polygon-Elemente ausgeben
//    std::cout << "First polygons: ";
//    for( i = 0; i < 10; ++i )
//    {
//    std::cout << polygons.at( i ) << " ";
//    }
//    std::cout << std::endl << "Last polygons: ";
//    for( i = polygons.size() - 10; i < polygons.size(); ++i )
//    {
//    std::cout << polygons.at( i ) << " ";
//    }
//    std::cout << std::endl;

    //  dieser Aufruf compiliert, gibt aber einen Linker-Fehler aus (undefined reference)
    //  return boost::shared_ptr< WTriangleMesh >( new WTriangleMesh( positions, polygons ) );  // nach Vorbild von WReaderELC.cpp

    //  einfacherer Konstruktor-Aufruf WTriangleMesh::WTriangleMesh( size_t vertNum, size_t triangleNum )
    //  funktioniert ebenso nicht:
    //  WTriangleMesh mesh( numPositions , numPolygons );

    //  genauso wenig dieser Test
//      size_t a = 3, b = 1;
//      WTriangleMesh mesh(a,b);
      //return mesh;
}

