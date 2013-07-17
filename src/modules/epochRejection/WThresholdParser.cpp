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

#include "core/common/WLogger.h"

#include "WThresholdParser.h"

/**
 * Constructor
 */
WThresholdParser::WThresholdParser()
{

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
 * \retrun
 *          return true, when the parsing was successful, else false.
 */
bool WThresholdParser::parse(std::string fname)
{
    this->init(); // init the parser.

    std::map<std::string,double> values;
    const char separator = ' ';
    std::ifstream fstream;  // file-Handle
    std::string line;
    size_t lineCount = 0;

    wlog::debug( CLASS ) << "start parsing: " << fname;

    // check whether or not the file is a vaild .cfg file.
    if(fname.find(".cfg") ==  std::string::npos)
    {
        wlog::debug( CLASS ) << "invalid file";
        return false;
    }

    // open the given file
    fstream.open(fname.c_str(), std::ifstream::in);

    if(!fstream || fstream.bad()) // test the file status
        wlog::debug( CLASS ) << "file not open";

    wlog::debug( CLASS ) << "start reading file";

    while ( fstream.good() ) // while find data
    {
        getline(fstream, line); // get next line from file

        lineCount++;

        if(isValidLine(line)) // test read line
        {
            if(line.find_first_of(separator) != std::string::npos)
            {
                // split the line at the separators position
                std::string label = line.substr(0,line.find_first_of(separator));
                std::string value = line.substr(line.find_first_of(separator) + 1);

                // insert the value and label to the map
                values.insert(std::map<std::string,double>::value_type(label,fromString<double>(value)));
            }
        }
    }

    wlog::debug( CLASS ) << "file closed: " << lineCount << " lines read.";

    this->m_thresholds = values; // assign values to the global member.

    return true;
}

/**
 * Method to return the threshold list.
 */
std::map<std::string,double> WThresholdParser::getThresholds()
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
bool WThresholdParser::isValidLine(std::string line)
{
    size_t i;
    const size_t patternsize = 4;

    std::string pattern[patternsize];
    pattern[0] = "gradReject";
    pattern[1] = "magReject";
    pattern[2] = "eegReject";
    pattern[3] = "eogReject";

    for(i = 0; i < patternsize; i++) // test string for all pattern
    {
        if(line.find(pattern[i]) != std::string::npos)
            return true; // one pattern matched
    }

    return false; // no match
}

/**
 * Method to convert a string in to the given data type.
 */
template<class T> T WThresholdParser::fromString(const std::string& s)
{
     std::istringstream stream (s);
     T t;
     stream >> t;
     return t;
}

/**
 * Class name.
 */
const std::string WThresholdParser::CLASS = "WThresholdParser";
