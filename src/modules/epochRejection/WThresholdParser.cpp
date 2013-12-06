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

#include <fstream>
#include <map>
#include <string>
#include <stdio.h>
#include <exception>

#include <boost/foreach.hpp>

#include "core/common/WLogger.h"
#include "core/data/WLEMMEnumTypes.h"

#include "WThresholdParser.h"
#include "WThreshold.h"

/**
 * Constructor
 */
WThresholdParser::WThresholdParser()
{
    m_list.reset( new std::list< WThreshold >() );

    m_patterns.reset( new std::map< std::string, LaBP::WEModalityType::Enum >() );
    m_patterns->insert( ModiMap::value_type( MODALITY_EEG, LaBP::WEModalityType::EEG ) );
    m_patterns->insert( ModiMap::value_type( MODALITY_EOG, LaBP::WEModalityType::EOG ) );
    m_patterns->insert( ModiMap::value_type( MODALITY_MEG_GRAD, LaBP::WEModalityType::MEG ) );
    m_patterns->insert( ModiMap::value_type( MODALITY_MEG_MAG, LaBP::WEModalityType::MEG ) );
}

/**
 * Desctructor
 */
WThresholdParser::~WThresholdParser()
{

}

/**
 * Method to parse a given .cfg file an return the containing threshold values.
 *
 * \param fname
 *          The file name to the thresholds.
 * \return
 *          return true, when the parsing was successful, else false.
 */
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

                    // insert the value and label to the map
                    values.insert( std::map< std::string, double >::value_type( label, fromString< double >( value ) ) );

                    /*
                     boost::shared_ptr< WThreshold > threshold;

                     switch( m_patterns->find( label )->second )
                     {
                     case LaBP::WEModalityType::EEG:
                     threshold.reset( new WThreshold( LaBP::WEModalityType::EEG, ::atof( value.c_str() ) ) );
                     break;
                     case LaBP::WEModalityType::EOG:
                     threshold.reset( new WThreshold( LaBP::WEModalityType::EOG, ::atof( value.c_str() ) ) );
                     break;
                     case LaBP::WEModalityType::MEG:
                     if( label == MODALITY_MEG_GRAD )
                     {
                     threshold.reset(
                     new WThresholdMEG( LaBP::WEGeneralCoilType::GRADIOMETER,
                     ::atof( value.c_str() ) ) );
                     }
                     else
                     if( label == MODALITY_MEG_MAG )
                     {
                     threshold.reset(
                     new WThresholdMEG( LaBP::WEGeneralCoilType::MAGNETOMETER,
                     ::atof( value.c_str() ) ) );
                     }
                     break;
                     default:
                     break;
                     }

                     m_list->push_back( *threshold.get() ); // add threshold to the list
                     */
                }
            }
        }

        wlog::debug( CLASS ) << "file closed: " << lineCount << " lines read.";

        this->m_thresholds = values; // assign values to the global member.

        rc = true;
    }
    catch( std::exception& e )
    {
        wlog::debug( CLASS ) << "error happened during parsing: " << e.what();
    }

    return rc;
}

/**
 * Method to return the threshold list.
 */
std::map< std::string, double > WThresholdParser::getThresholds()
{
    return this->m_thresholds;
}

/**
 * Method to reset members before processing.
 */
void WThresholdParser::init()
{
    this->m_thresholds.clear();
}

/**
 * Method to define whether or not a line has to parse.
 */
bool WThresholdParser::isValidLine( std::string line )
{
    BOOST_FOREACH(ModiMap::value_type it , *m_patterns.get())
    {
        if( line.find( it.first ) != std::string::npos )
            return true;
    }

    return false;

    /*
     size_t i;
     const size_t patternsize = 4;

     std::string pattern[patternsize];
     pattern[0] = "gradReject";
     pattern[1] = "magReject";
     pattern[2] = "eegReject";
     pattern[3] = "eogReject";

     for( i = 0; i < patternsize; i++ ) // test string for all pattern
     {
     if( line.find( pattern[i] ) != std::string::npos )
     return true; // one pattern matched
     }

     return false; // no match
     */
}

/**
 * Method to convert a string in to the given data type.
 */
template< class T > T WThresholdParser::fromString( const std::string& s )
{
    std::istringstream stream( s );
    T t;
    stream >> t;
    return t;
}

boost::shared_ptr< std::list< WThreshold > > WThresholdParser::getThresholdList() const
{
    return m_list;
}

/**
 * Class name.
 */
const std::string WThresholdParser::CLASS = "WThresholdParser";

/**
 * Modality type patterns.
 */
const std::string WThresholdParser::MODALITY_EEG = "eegReject";
const std::string WThresholdParser::MODALITY_EOG = "eogReject";
const std::string WThresholdParser::MODALITY_MEG_GRAD = "gradReject";
const std::string WThresholdParser::MODALITY_MEG_MAG = "magReject";
