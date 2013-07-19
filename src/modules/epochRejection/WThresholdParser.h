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

#ifndef WTHRESHOLDPARSER_H_
#define WTHRESHOLDPARSER_H_

#include <boost/shared_ptr.hpp>

#include <map>
#include <string>

class WThresholdParser
{
public:

    static const std::string CLASS;

    typedef boost::shared_ptr< WThresholdParser > SPtr;

    /**
     * Constructor
     */
    WThresholdParser();

    /**
     * Destructor
     */
    virtual ~WThresholdParser();

    /**
     * Method to parse a given .cfg file an return the containing threshold values.
     */
    bool parse(std::string fname);

    /**
     * Method to retrun the parsed values as a list of double values.
     */
    std::map<std::string,double> getThresholds();

private:

    /**
     * Method to init the parser.
     */
    void init();

    /**
     * This method searches in the given line for defined string patterns.
     *
     * \return true if a pattern was matched, else false.
     */
    bool isValidLine(std::string line);

    /**
     * A Map with string as key and double as value, which contains the thresholds after parsing.
     */
    std::map<std::string,double> m_thresholds;

    /**
     * This method converts a string into the given type, specified in the type parameter T.
     */
    template<class T> T fromString(const std::string& s);
};

#endif /* WTHRESHOLDPARSER_H_ */
