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

#include <fstream>
#include <list>
#include <map>
#include <string>
#include <stdio.h>
#include <exception>

#include <boost/foreach.hpp>

#include "core/common/WLogger.h"
#include "core/data/enum/WLEModality.h"

#include "WThresholdParser.h"
#include "WThreshold.h"

WThresholdParser::WThresholdParser()
{
    m_list.reset( new std::list< WThreshold >() );

    m_patterns.reset( new std::map< std::string, WLEModality::Enum >() );
    m_patterns->insert( ModiMap::value_type( MODALITY_EEG, WLEModality::EEG ) );
    m_patterns->insert( ModiMap::value_type( MODALITY_EOG, WLEModality::EOG ) );
    m_patterns->insert( ModiMap::value_type( MODALITY_MEG_GRAD, WLEModality::MEG_GRAD ) );
    m_patterns->insert( ModiMap::value_type( MODALITY_MEG_MAG, WLEModality::MEG_MAG ) );
}

WThresholdParser::~WThresholdParser()
{
}

bool WThresholdParser::parse( std::string fname )
{
    this->init(); // init the parser.

    std::map< std::string, double > values;
    const char separator = ' ';
    std::ifstream fstream;  // file-Handle
    std::string line;
    size_t lineCount = 0;
    bool rc = false;

    wlog::debug( CLASS ) << "start parsing: " << fname;

    // check whether or not the file is a vaild .cfg file.
    if( fname.find( ".cfg" ) == std::string::npos )
    {
        wlog::debug( CLASS ) << "invalid file";
        return rc;
    }

    try
    {
        // open the given file
        fstream.open( fname.c_str(), std::ifstream::in );

        if( !fstream || fstream.bad() )  // test the file status
            wlog::debug( CLASS ) << "file not open";

        wlog::debug( CLASS ) << "start reading file";

        while( fstream.good() )  // while find data
        {
            getline( fstream, line ); // get next line from file

            lineCount++;

            if( isValidLine( line ) ) // test read line
            {
                if( line.find_first_of( separator ) != std::string::npos )
                {
                    // split the line at the separators position
                    std::string label = line.substr( 0, line.find_first_of( separator ) );
                    std::string value = line.substr( line.find_first_of( separator ) + 1 );

                    if( !m_patterns->count( label ) )
                        continue;

                    WThreshold threshold( m_patterns->at( label ), ::atof( value.c_str() ) );
                    m_list->push_back( threshold ); // add threshold to the list
                }
            }
        }

        wlog::debug( CLASS ) << "file closed: " << lineCount << " lines read.";

        rc = true;

        fstream.close();
    }
    catch( const std::exception& e )
    {
        fstream.close();

        wlog::error( CLASS ) << "error happened during parsing: " << e.what();
    }

    return rc;
}

void WThresholdParser::init()
{
    this->m_list->clear();
}

bool WThresholdParser::isValidLine( std::string line )
{
    BOOST_FOREACH( ModiMap::value_type it , *m_patterns.get() )
    {
        if( line.find( it.first ) != std::string::npos )
        {
            return true; // match found
        }
    }

    return false; // no match
}

boost::shared_ptr< std::list< WThreshold > > WThresholdParser::getThresholdList() const
{
    return m_list;
}

const std::string WThresholdParser::CLASS = "WThresholdParser";

const std::string WThresholdParser::MODALITY_EEG = "eegReject";
const std::string WThresholdParser::MODALITY_EOG = "eogReject";
const std::string WThresholdParser::MODALITY_MEG_GRAD = "gradReject";
const std::string WThresholdParser::MODALITY_MEG_MAG = "magReject";
