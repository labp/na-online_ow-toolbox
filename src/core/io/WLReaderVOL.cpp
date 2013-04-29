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

#include <core/common/exceptions/WTypeMismatch.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/common/WLogger.h>
#include <core/common/WStringUtils.h>

#include "core/dataHandler/WDataSetEMMSubject.h"
#include "core/dataHandler/WDataSetEMMBemBoundary.h"

#include "WLReaderBND.h"
#include "WLReaderVOL.h"

using namespace LaBP;
using namespace std;

const string CLASS = "WLReaderVOL";

WLReaderVOL::WLReaderVOL( std::string fname ) :
                WLReader( fname )
{
    wlog::debug( CLASS ) << "file: " << fname;
}

WLReaderVOL::ReturnCode::Enum WLReaderVOL::read(
                boost::shared_ptr< std::vector< boost::shared_ptr< WDataSetEMMBemBoundary > > > boundaries )
{
    ifstream ifs;
    ifs.open( m_fname.c_str(), ifstream::in );

    if( !ifs || ifs.bad() )
    {
        return ReturnCode::ERROR_FOPEN;
    }

    string line;
    size_t countBnd = 0;
    bool hasNumBnd = false, hasUnit = false, hasConduct = false;
    ReturnCode::Enum rc = ReturnCode::ERROR_UNKNOWN;
    try
    {
        while( getline( ifs, line ) && ( !hasNumBnd || !hasUnit || !hasConduct ) )
        {
            if( line.find( "NumberBoundaries=" ) == 0 )
            {
                rc = readNumBoundaries( line, countBnd );
                if( !boundaries )
                {
                    boundaries.reset( new vector< boost::shared_ptr< WDataSetEMMBemBoundary > >() );
                }
                boundaries->reserve( countBnd );
                for( size_t i = 0; i < countBnd; ++i )
                {
                    boundaries->push_back( boost::shared_ptr< WDataSetEMMBemBoundary >( new WDataSetEMMBemBoundary() ) );
                }
                hasNumBnd = true;
            }
            else
                if( line.find( "UnitConduct=" ) == 0 )
                {
                    WEUnit::Enum unit = WEUnit::UNKNOWN_UNIT;
                    rc = readConductUnit( line, unit );
                    hasUnit = true;
                }
                else
                    if( line.find( "Conductivities" ) == 0 )
                    {
                        rc = readConductivities( ifs, *boundaries );
                        hasConduct = true;
                    }
                    else
                        if( line.find( "Boundary" ) == 0 )
                        {
                            rc = readBndFiles( ifs, line, *boundaries );
                        }
        }
    }
    catch( WTypeMismatch& e )
    {
        wlog::error( CLASS ) << e.what();
        rc = ReturnCode::ERROR_UNKNOWN;
    }

    ifs.close();
    return rc;
}

WLReaderVOL::ReturnCode::Enum WLReaderVOL::readNumBoundaries( std::string& line, size_t& count )
{
    vector< string > tokens = string_utils::tokenize( line );
    count = string_utils::fromString< size_t >( tokens.at( 1 ) );
    wlog::debug( CLASS ) << "Number of boundaries: " << count;
    return ReturnCode::SUCCESS;
}

WLReaderVOL::ReturnCode::Enum WLReaderVOL::readConductUnit( std::string& line, WEUnit::Enum& unit )
{
    vector< string > tokens = string_utils::tokenize( line );
    string sunit = tokens.at( 1 );
    wlog::debug( CLASS ) << "Unit: " << sunit;
    if( sunit.find( "S/m" ) != string::npos )
    {
        unit = WEUnit::SIEMENS_PER_METER;
        return ReturnCode::SUCCESS;
    }
    else
    {
        unit = WEUnit::UNKNOWN_UNIT;
        wlog::warn( CLASS ) << "Unknown unit.";
        return ReturnCode::ERROR_UNKNOWN;
    }
}

WLReaderVOL::ReturnCode::Enum WLReaderVOL::readConductivities( std::ifstream& ifs,
                std::vector< boost::shared_ptr< WDataSetEMMBemBoundary > >& boundaries )
{
    if( boundaries.size() == 0 )
    {
        wlog::error( CLASS ) << "Empty boundary vector!";
        return ReturnCode::ERROR_UNKNOWN;
    }

    string line;
    getline( ifs, line );
    if( !ifs.good() )
    {
        return ReturnCode::ERROR_FREAD;
    }

    vector< string > lineTokens = string_utils::tokenize( line, " " );
    float conduct;
    for( vector< string >::size_type i = 0; i < lineTokens.size() && i < boundaries.size(); ++i )
    {
        conduct = string_utils::fromString< float >( lineTokens.at( i ) );
        boundaries.at( i )->setConductivity( conduct );
    }

    return ReturnCode::SUCCESS;
}

WLReaderVOL::ReturnCode::Enum WLReaderVOL::readBndFiles( std::ifstream& ifs, string& line,
                std::vector< boost::shared_ptr< WDataSetEMMBemBoundary > >& boundaries )
{
    if( boundaries.size() == 0 )
    {
        wlog::error( CLASS ) << "Empty boundary vector!";
        return ReturnCode::ERROR_UNKNOWN;
    }

    size_t count = 0;

    string path = m_fname.substr( 0, m_fname.find_last_of( '/' ) + 1 );

    do
    {
        ++count;
        vector< string > tokens = string_utils::tokenize( line );
        string fname = tokens.at( 1 );
        WLReaderBND reader( path + fname );
        if( reader.read( boundaries.at( count - 1 ) ) != WLReaderBND::ReturnCode::SUCCESS )
        {
            wlog::error( CLASS ) << "Error while reading " << fname;
        }
        getline( ifs, line );
        if( !ifs.good() && !ifs.eof() )
        {
            wlog::error( CLASS ) << "Unexpected file end!";
            return ReturnCode::ERROR_FREAD;
        }
    } while( count < boundaries.size() );

    return ReturnCode::SUCCESS;
}
