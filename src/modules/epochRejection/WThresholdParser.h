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

#ifndef WTHRESHOLDPARSER_H_
#define WTHRESHOLDPARSER_H_

#include <map>
#include <string>
#include <list>

#include <boost/shared_ptr.hpp>

#include "WThreshold.h"

class WThresholdParser
{
public:
    /**
     * Class name.
     */
    static const std::string CLASS;

    /**
     * Label for the EEG modality.
     */
    static const std::string MODALITY_EEG;

    /**
     * Label for the EOG modality.
     */
    static const std::string MODALITY_EOG;

    /**
     * Label for the MEG gradiometer modality.
     */
    static const std::string MODALITY_MEG_GRAD;

    /**
     * Label for the MEG magnetometer modality.
     */
    static const std::string MODALITY_MEG_MAG;

    /**
     * A pointer on the class.
     */
    typedef boost::shared_ptr< WThresholdParser > SPtr;

    /**
     * A map of string and modality enum.
     */
    typedef std::map< std::string, WLEModality::Enum > ModiMap;

    /**
     * A pointer on a modality map.
     */
    typedef boost::shared_ptr< ModiMap > ModiMapSPtr;

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
     *
     * \param fname
     *          The file name to the thresholds.
     * \return
     *          return true, when the parsing was successful, else false.
     */
    bool parse( std::string fname );

    /**
     * Gets the parsed thresholds as pointer on a list.
     *
     * @return A shared pointer in the parsed list.
     */
    boost::shared_ptr< std::list< WThreshold > > getThresholdList() const;

private:
    /**
     * Method to reset members before processing.
     */
    void init();

    /**
     * This method searches in the given line for defined string patterns.
     *
     * \return true if a pattern was matched, else false.
     */
    bool isValidLine( std::string line );

    /**
     * A list of threshold objects.
     */
    boost::shared_ptr< std::list< WThreshold > > m_list;

    /**
     * Map to match the string labels to the appropriate modality enum.
     */
    ModiMapSPtr m_patterns;
};

#endif  // WTHRESHOLDPARSER_H_
