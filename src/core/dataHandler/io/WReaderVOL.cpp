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
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/common/WStringUtils.h>
#include <core/dataHandler/exceptions/WDHIOFailure.h>
#include <core/dataHandler/exceptions/WDHNoSuchFile.h>
#include <core/dataHandler/exceptions/WDHParseError.h>
#include <core/dataHandler/WDataSet.h>

#include "WReaderVOL.h"
#include "WReaderBND.h"

WReaderVOL::WReaderVOL( std::string fname )
    : WReader( fname )
{
}

std::vector< std::string > WReaderVOL::read()
{
    std::ifstream ifs;
    ifs.open( m_fname.c_str(), std::ifstream::in );
    if( !ifs || ifs.bad() )
    {
        throw WDHNoSuchFile( std::string( "Problem loading file " + m_fname + ". Probably file not found." ) );
    }

    // get path name
    std::string pathToFile;
    std::vector< std::string > pathNameTokens = string_utils::tokenize( m_fname, "/" );

    // remove last element (file name); there's probably a more elegant way
    pathNameTokens.erase( pathNameTokens.begin() + pathNameTokens.size() - 1 );
    pathToFile = "";

    for(int i = 0; i < pathNameTokens.size() ; ++i )
    {
    pathToFile += pathNameTokens.at( i );
    pathToFile += "/";
    }

    std::string line;
    while( ifs.good() && line.substr( 0, 16 ) != "NumberBoundaries" )  // go to number of boundaries
    {
        std::getline( ifs, line );
        if( !ifs.good() )
        {
            throw WDHIOFailure( std::string( "Unexpected end of .vol file " + m_fname ) );
        }
    }

    std::vector< std::string > lineTokens = string_utils::tokenize( line );
    unsigned int numBoundaries = string_utils::fromString< unsigned int >( lineTokens.at( 1 ) );
    unsigned int i = 0;

    std::vector< double > conductivities;
    conductivities.reserve( numBoundaries );

    while( ifs.good() && line.substr( 0, 14 ) != "Conductivities" )
    {
        std::getline( ifs, line );
        if( !ifs.good() )
        {
            throw WDHIOFailure( std::string( "Unexpected end of .vol file " + m_fname ) );
        }
    }


    std::getline( ifs, line );
    if( !ifs.good() )
    {
        throw WDHIOFailure( std::string( "Unexpected end of .vol file " + m_fname ) );
    }
    else
    {
    // read conductivities
        lineTokens = string_utils::tokenize( line );
        for( i = 0; i < lineTokens.size(); ++i )
        {
            double cond = string_utils::fromString< double >( lineTokens.at( i ) );
            conductivities.push_back( cond );
        }
    }

    while( ifs.good() )
    {
    while( line.substr( 0, 8 )  != "Boundary" )
    {
        std::getline( ifs, line );
        if( !ifs.good() )
        {
        throw WDHIOFailure( std::string( "Unexpected end of .vol file " + m_fname ) );
        }
    }

    lineTokens = string_utils::tokenize( line );
    filenameBoundaries.push_back( pathToFile + lineTokens.at( 1 ) );
    std::getline( ifs, line );
    }

    if( numBoundaries != filenameBoundaries.size() )
    {
    throw WDHParseError( std::string( "Number of boundaries and actual number of referenced boundary-files (.bnd-files) in .vol-file don't match" + m_fname ) );
    }
    ifs.close();

    std::cout << "Boundaries:     " << numBoundaries << std::endl;
    for( i = 0; i < numBoundaries; ++i)
    {
    std::cout << "File " << i << ": " << filenameBoundaries.at( i ) << std::endl;
    std::cout << "Conductivity " << i << ": " << conductivities.at( i ) << std::endl;
    }



//    for( i = 0; i < numBoundaries; ++i )
//    {
//     WReaderBND bndReader( pathToFile + filenameBoundaries.at( i ) );
//     WTriangleMesh mesh1 =bndReader.read();
//    }

    return filenameBoundaries;
}
