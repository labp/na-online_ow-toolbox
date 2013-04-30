// TODO doc & license

#ifndef WLREADEREXPERIMENT_H_
#define WLREADEREXPERIMENT_H_

#include <set>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>

#include <core/dataHandler/exceptions/WDHNoSuchFile.h>

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

    static boost::filesystem::path getExperimentRootFromFiff(boost::filesystem::path fiffFile);
    static std::string getSubjectFromFiff(boost::filesystem::path fiffFile);
    static std::string getTrialFromFiff(boost::filesystem::path fiffFile);

    // BEM Layer //
    std::set< std::string > findBems();
    bool readBem( std::string fname, LaBP::WLEMMSubject::SPtr subject ); // TODO evt return code

    // Source Space //
    std::set< std::string > findSurfaceKinds(); // TODO evt. Enum als Rückgabe
    bool readSourceSpace( std::string surfaceKind, LaBP::WLEMMSubject::SPtr subject ); // TODO evt return code

    // Leadfield //
    std::set< std::string > findLeadfieldTrials();
    bool readLeadFields( std::string surface, std::string bemName, std::string trial, LaBP::WLEMMSubject::SPtr subject );
    bool readLeadField( std::string surface, std::string bemName, std::string trial, std::string modality,
                    LaBP::WLEMMSubject::SPtr subject );

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
    static const std::string m_EEG;
    static const std::string m_MEG;
    static const std::string m_VOL;
    static const std::string m_DIP;
    static const std::string m_MAT;
    static const std::string CLASS;
};

#endif /* WLREADEREXPERIMENT_H_ */
