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

#ifndef WLREADEREXPERIMENT_H_
#define WLREADEREXPERIMENT_H_

#include <set>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>

#include <core/dataHandler/exceptions/WDHNoSuchFile.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMSubject.h"

class WLReaderExperiment
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLReaderExperiment > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLReaderExperiment > ConstSPtr;

    WLReaderExperiment( std::string experimentPath, std::string subject ) throw( WDHNoSuchFile );
    virtual ~WLReaderExperiment();

    static boost::filesystem::path getExperimentRootFromFiff( boost::filesystem::path fiffFile );
    static std::string getSubjectFromFiff( boost::filesystem::path fiffFile );
    static std::string getTrialFromFiff( boost::filesystem::path fiffFile );

    // BEM Layer //
    std::set< std::string > findBems();
    bool readBem( std::string fname, WLEMMSubject::SPtr subject ); // TODO evt return code

    // Source Space //
    std::set< std::string > findSurfaceKinds(); // TODO evt. Enum als Rückgabe
    bool readSourceSpace( std::string surfaceKind, WLEMMSubject::SPtr subject ); // TODO evt return code

    // Leadfield //
    std::set< std::string > findLeadfieldTrials();
    bool readLeadFields( std::string surface, std::string bemName, std::string trial, WLEMMSubject::SPtr subject );
    bool readLeadField( std::string surface, std::string bemName, std::string trial, std::string modality,
                    WLEMMSubject::SPtr subject );
    bool readLeadFieldMat( const std::string& fName, WLMatrix::SPtr& matrix );
    bool readLeadFieldFiff( const std::string& fName, WLMatrix::SPtr& matrix );

private:
    // Initial information //
    const std::string m_PATH_EXPERIMENT;
    const std::string m_SUBJECT;

    // Known folder names //
    static const std::string m_FOLDER_BEM;
    static const std::string m_FOLDER_FSLVOL;
    static const std::string m_FOLDER_RESULTS;
    static const std::string m_FOLDER_SURF;

    // Known file name parts //
    static const std::string m_PIAL;
    static const std::string m_INFLATED;
    static const std::string m_LEADFIELD;
    static const std::string m_LH;
    static const std::string m_RH;
    static const std::string m_LHRH;
    static const std::string m_EEG;
    static const std::string m_MEG;
    static const std::string m_VOL;
    static const std::string m_DIP;
    static const std::string m_MAT;
    static const std::string m_FIFF;
    static const std::string CLASS;
};

#endif /* WLREADEREXPERIMENT_H_ */
