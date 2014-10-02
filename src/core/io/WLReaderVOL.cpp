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
#include <fstream>
#include <list>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/exceptions/WTypeMismatch.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/common/WLogger.h>
#include <core/common/WStringUtils.h>

#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMBemBoundary.h"

#include "WLReaderBND.h"
#include "WLReaderVOL.h"

using std::ifstream;
using std::list;
using std::string;
using std::vector;

const std::string WLReaderVOL::CLASS = "WLReaderVOL";

WLReaderVOL::WLReaderVOL( std::string fname ) :
                WLReaderGeneric< std::list< WLEMMBemBoundary::SPtr > >( fname )
{
    wlog::debug( CLASS ) << "file: " << fname;
}

WLIOStatus::IOStatusT WLReaderVOL::read( std::list< WLEMMBemBoundary::SPtr >* const boundaries )
{
    ifstream ifs;
    ifs.open( m_fname.c_str(), ifstream::in );

    if( !ifs || ifs.bad() )
    {
        return WLIOStatus::ERROR_FOPEN;
    }

    string line;
    size_t countBnd = 0;
    bool hasNumBnd = false, hasUnit = false, hasConduct = false;
    WLIOStatus::IOStatusT rc = WLIOStatus::ERROR_UNKNOWN;
    try
    {
        while( getline( ifs, line ) && ( !hasNumBnd || !hasUnit || !hasConduct ) )
        {
            if( line.find( "NumberBoundaries=" ) == 0 )
            {
                rc = readNumBoundaries( &countBnd, line );
                for( size_t i = 0; i < countBnd; ++i )
                {
                    boundaries->push_back( boost::shared_ptr< WLEMMBemBoundary >( new WLEMMBemBoundary() ) );
                }
                hasNumBnd = true;
            }
            else
                if( line.find( "UnitConduct=" ) == 0 )
                {
                    WLEUnit::Enum unit = WLEUnit::NONE;
                    rc = readConductUnit( &unit, line );
                    hasUnit = true;
                }
                else
                    if( line.find( "Conductivities" ) == 0 )
                    {
                        rc = readConductivities( ifs, boundaries );
                        hasConduct = true;
                    }
                    else
                        if( line.find( "Boundary" ) == 0 )
                        {
                            rc = readBndFiles( ifs, &line, boundaries );
                        }
        }
    }
    catch( const WTypeMismatch& e )
    {
        wlog::error( CLASS ) << e.what();
        rc = WLIOStatus::ERROR_UNKNOWN;
    }

    ifs.close();
    return rc;
}

WLIOStatus::IOStatusT WLReaderVOL::readNumBoundaries( size_t* const count, const std::string& line )
{
    vector< string > tokens = string_utils::tokenize( line );
    *count = string_utils::fromString< size_t >( tokens.at( 1 ) );
    wlog::debug( CLASS ) << "Number of boundaries: " << count;
    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderVOL::readConductUnit( WLEUnit::Enum* const unit, const std::string& line )
{
    vector< string > tokens = string_utils::tokenize( line );
    string sunit = tokens.at( 1 );
    wlog::debug( CLASS ) << "Unit: " << sunit;
    if( sunit.find( "S/m" ) != string::npos )
    {
        *unit = WLEUnit::SIEMENS_PER_METER;
        return WLIOStatus::SUCCESS;
    }
    else
    {
        *unit = WLEUnit::NONE;
        wlog::warn( CLASS ) << "Unknown unit.";
        return WLIOStatus::ERROR_UNKNOWN;
    }
}

WLIOStatus::IOStatusT WLReaderVOL::readConductivities( std::ifstream& ifs, std::list< WLEMMBemBoundary::SPtr >* const boundaries )
{
    if( boundaries->size() == 0 )
    {
        wlog::error( CLASS ) << "Empty boundary vector!";
        return WLIOStatus::ERROR_UNKNOWN;
    }

    string line;
    getline( ifs, line );
    if( !ifs.good() )
    {
        return WLIOStatus::ERROR_FREAD;
    }

    vector< string > lineTokens = string_utils::tokenize( line, " " );
    vector< string >::iterator cit = lineTokens.begin();
    list< WLEMMBemBoundary::SPtr >::iterator bit = boundaries->begin();
    for( ; cit != lineTokens.end() && bit != boundaries->end(); ++cit, ++bit )
    {
        const float conduct = string_utils::fromString< float >( *cit );
        ( *bit )->setConductivity( conduct );
    }

    return WLIOStatus::SUCCESS;
}

WLIOStatus::IOStatusT WLReaderVOL::readBndFiles( std::ifstream& ifs, string* const line,
                std::list< WLEMMBemBoundary::SPtr >* const boundaries )
{
    if( boundaries->size() == 0 )
    {
        wlog::error( CLASS ) << "Empty boundary vector!";
        return WLIOStatus::ERROR_UNKNOWN;
    }

    string path = m_fname.substr( 0, m_fname.find_last_of( '/' ) + 1 );

    list< WLEMMBemBoundary::SPtr >::iterator bit;
    for( bit = boundaries->begin(); bit != boundaries->end(); ++bit )
    {
        vector< string > tokens = string_utils::tokenize( *line );
        string fname = tokens.at( 1 );
        WLReaderBND reader( path + fname );
        if( reader.read( *bit ) != WLReaderBND::ReturnCode::SUCCESS )
        {
            wlog::error( CLASS ) << "Error while reading " << fname;
        }
        getline( ifs, *line );
        if( !ifs.good() && !ifs.eof() )
        {
            wlog::error( CLASS ) << "Unexpected file end!";
            return WLIOStatus::ERROR_FREAD;
        }
    }

    return WLIOStatus::SUCCESS;
}
