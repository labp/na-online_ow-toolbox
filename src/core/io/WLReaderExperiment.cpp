// TODO doc & license
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>

#include <core/common/WAssert.h>
#include <core/common/WIOTools.h>
#include <core/common/WLogger.h>
#include <core/dataHandler/exceptions/WDHNoSuchFile.h>

#include "core/data/WLEMMBemBoundary.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMSurface.h"

#include "WLReaderDIP.h"
#include "WLReaderMatMab.h"
#include "WLReaderVOL.h"

#include "WLReaderExperiment.h"

using namespace boost::filesystem;
using namespace LaBP;
using namespace std;

const string WLReaderExperiment::m_FOLDER_BEM = "bem";
const string WLReaderExperiment::m_FOLDER_FSLVOL = "fslvol";
const string WLReaderExperiment::m_FOLDER_RESULTS = "results";
const string WLReaderExperiment::m_FOLDER_SURF = "surf";

const string WLReaderExperiment::m_PIAL = "pial";
const string WLReaderExperiment::m_INFLATED = "inflated";
const string WLReaderExperiment::m_LEADFIELD = "leadfield";
const string WLReaderExperiment::m_LH = "lh";
const string WLReaderExperiment::m_RH = "rh";
const string WLReaderExperiment::m_EEG = "eeg";
const string WLReaderExperiment::m_MEG = "meg";
const string WLReaderExperiment::m_VOL = ".vol";
const string WLReaderExperiment::m_DIP = ".dip";
const string WLReaderExperiment::m_MAT = ".mat";

const string WLReaderExperiment::CLASS = "WLReaderExperiment";

WLReaderExperiment::WLReaderExperiment( std::string experimentPath, std::string subject ) throw( WDHNoSuchFile ) :
                m_PATH_EXPERIMENT( experimentPath ), m_SUBJECT( subject )
{
    if( !fileExists( m_PATH_EXPERIMENT ) )
    {
        throw WDHNoSuchFile( m_PATH_EXPERIMENT );
    }
}

WLReaderExperiment::~WLReaderExperiment()
{
}

boost::filesystem::path WLReaderExperiment::getExperimentRootFromFiff( boost::filesystem::path fiffFile )
{
    wlog::debug( CLASS ) << "fileName: " << fiffFile.filename();
    // TODO maybe exception if first parent_path is filesystem root.
    boost::filesystem::path expPath = fiffFile.parent_path().parent_path().parent_path();
    size_t isExperiment = 0;

    if( !boost::filesystem::exists( expPath ) )
    {
        return fiffFile;
    }

    directory_iterator end_it;
    for( boost::filesystem::directory_iterator it( expPath ); it != end_it; ++it )
    {
        if( isExperiment >= 2 )
        {
            break;
        }
        if( boost::filesystem::is_directory( it->status() ) && it->path().filename() == m_FOLDER_RESULTS )
        {
            ++isExperiment;
        }
        else
            if( boost::filesystem::is_directory( it->status() ) && it->path().filename() == m_FOLDER_FSLVOL )
            {
                ++isExperiment;
            }
    }

    if( isExperiment >= 2 )
    {
        wlog::debug( CLASS ) << "Experiment root: " << expPath.filename();
        return expPath;
    }
    else
    {
        wlog::error( CLASS ) << "Could not retrieve experiment root!";
        return fiffFile;
    }
}

std::string WLReaderExperiment::getSubjectFromFiff( boost::filesystem::path fiffFile )
{
    boost::regex re( "(\\D+\\d+)(\\D+)(\\d+).fif" );
    boost::cmatch matches;
    std::string fname = fiffFile.filename().string();
    std::string subject;
    if( boost::regex_match( fname.c_str(), matches, re ) )
    {
        // i = 0 contains complete string
        if( 0 < matches.size() )
        {
            // first and second are the start and end of the string
            subject = std::string( matches[1].first, matches[1].second );
        }
    }
    else
    {
        wlog::error( CLASS ) << "No subject found!";
    }

    return subject;
}

std::string WLReaderExperiment::getTrialFromFiff( boost::filesystem::path fiffFile )
{
    boost::regex re( "\\D+(\\d+)(\\D+)(\\d+).fif" );
    boost::cmatch matches;
    std::string fname = fiffFile.filename().string();
    std::string trial;
    if( boost::regex_match( fname.c_str(), matches, re ) )
    {
        // i = 0 contains complete string
        if( 1 < matches.size() )
        {
            // first and second are the start and end of the string
            trial = std::string( matches[2].first, matches[2].second );
        }
    }
    else
    {
        wlog::error( CLASS ) << "No trial found!";
    }

    return trial;
}

std::set< std::string > WLReaderExperiment::findBems()
{
    wlog::debug( CLASS ) << "findBems() called!";
    path path( m_PATH_EXPERIMENT );
    path /= m_FOLDER_FSLVOL;
    path /= m_SUBJECT;
    path /= m_FOLDER_BEM;

    std::set< std::string > volFiles;
    if( !exists( path ) )
    {
        wlog::error( CLASS ) << path.string() << " does not exist!";
        return volFiles;
    }

    directory_iterator end;
    for( directory_iterator it( path ); it != end; ++it )
    {
        if( is_regular_file( it->status() ) && it->path().extension() == m_VOL )
        {

            volFiles.insert( it->path().filename().string() );
        }
    }

    wlog::debug( CLASS ) << "volFiles: " << volFiles.size();
    return volFiles;
}

bool WLReaderExperiment::readBem( std::string fname, LaBP::WLEMMSubject::SPtr subject )
{
    wlog::debug( CLASS ) << "readBem() called!";
    path path( m_PATH_EXPERIMENT );
    path /= m_FOLDER_FSLVOL;
    path /= m_SUBJECT;
    path /= m_FOLDER_BEM;
    path /= fname;

    boost::shared_ptr< vector< LaBP::WLEMMBemBoundary::SPtr > > bems( new vector< LaBP::WLEMMBemBoundary::SPtr >() );
    WLReaderVOL reader( path.string() );
    WLReaderVOL::ReturnCode::Enum rc = reader.read( bems );
    if( rc == WLReaderVOL::ReturnCode::SUCCESS )
    {
        subject->setBemBoundaries( bems );
        wlog::info( CLASS ) << "Read " << bems->size() << " BEM boundaries.";
        return true;
    }
    else
    {
        wlog::error( CLASS ) << "Read no BEM boundaries.";
        return false;
    }
}

std::set< std::string > WLReaderExperiment::findSurfaceKinds()
{
    wlog::debug( CLASS ) << "findSurfaceKinds() called!";
    path path( m_PATH_EXPERIMENT );
    path /= m_FOLDER_FSLVOL;
    path /= m_SUBJECT;
    path /= m_FOLDER_SURF;

    std::set< std::string > surfaces;
    if( !exists( path ) )
    {
        wlog::error( CLASS ) << path.string() << " does not exist!";
        return surfaces;
    }

    std::string tmp;
    directory_iterator end;
    for( directory_iterator it( path ); it != end; ++it )
    {
        if( is_regular_file( it->status() ) && it->path().extension() == m_DIP )
        {
            tmp = it->path().string();
            if( tmp.find( m_PIAL ) != std::string::npos )
            {
                surfaces.insert( m_PIAL );
            }
            if( tmp.find( m_INFLATED ) != std::string::npos )
            {
                surfaces.insert( m_INFLATED );
            }
        }
    }

    wlog::debug( CLASS ) << "surfaces: " << surfaces.size();
    return surfaces;
}

bool WLReaderExperiment::readSourceSpace( std::string surfaceKind, LaBP::WLEMMSubject::SPtr subject )
{
    wlog::debug( CLASS ) << "readBem() called!";
    bool rc = true;

    path folder( m_PATH_EXPERIMENT );
    folder /= m_FOLDER_FSLVOL;
    folder /= m_SUBJECT;
    folder /= m_FOLDER_SURF;

    path lhFile( folder );
    lhFile /= m_LH + "." + surfaceKind + m_DIP;
    wlog::info( CLASS ) << "Read file: " << lhFile.string();

    WLEMMSurface::SPtr lhSurface( new WLEMMSurface() );
    WLReaderDIP lhReader( lhFile.string() );
    WLReaderDIP::ReturnCode::Enum lhReturn = lhReader.read( lhSurface );
    if( lhReturn == WLReaderDIP::ReturnCode::SUCCESS )
    {
        wlog::info( CLASS ) << "Successfully read left surface!";
        lhSurface->setHemisphere( WLEMMSurface::Hemisphere::LEFT );
        subject->setSurface( lhSurface );
    }
    else
    {
        wlog::error( CLASS ) << "Could not load left surface!";
        rc &= false;
    }

    path rhFile( folder );
    rhFile /= m_RH + "." + surfaceKind + m_DIP;
    wlog::info( CLASS ) << "Read file: " << rhFile.string();

    WLEMMSurface::SPtr rhSurface( new WLEMMSurface() );
    WLReaderDIP rhReader( rhFile.string() );
    WLReaderDIP::ReturnCode::Enum rhReturn = rhReader.read( rhSurface );
    if( rhReturn == WLReaderDIP::ReturnCode::SUCCESS )
    {
        wlog::info( CLASS ) << "Successfully read right surface!";
        rhSurface->setHemisphere( WLEMMSurface::Hemisphere::RIGHT );
        subject->setSurface( rhSurface );
    }
    else
    {
        wlog::error( CLASS ) << "Could not load right surface!";
        rc &= false;
    }

    // Combine lh and rh
    if( !rc )
    {
        return rc;
    }
    wlog::info( CLASS ) << "Combine left and right surface to BOTH.";
    WLEMMSurface::SPtr bothSurface( new WLEMMSurface( *rhSurface ) );
    bothSurface->setHemisphere( WLEMMSurface::Hemisphere::BOTH );
    boost::shared_ptr< std::vector< WPosition > > bVertex = bothSurface->getVertex();
    std::vector< WVector3i >& bFaces = bothSurface->getFaces();

    bVertex->insert( bVertex->end(), lhSurface->getVertex()->begin(), lhSurface->getVertex()->end() );
    bVertex->insert( bVertex->end(), rhSurface->getVertex()->begin(), rhSurface->getVertex()->end() );
    bFaces.insert( bFaces.end(), lhSurface->getFaces().begin(), lhSurface->getFaces().end() );
    // NOTE: remind offset of faces
    const int tmp = static_cast< int >( lhSurface->getVertex()->size() );
    WVector3i offset( tmp, tmp, tmp );
    WVector3i face;
    for( std::vector< WVector3i >::const_iterator it = rhSurface->getFaces().begin(); it != rhSurface->getFaces().end(); ++it )
    {
        face = *it;
        face += offset;
        bFaces.push_back( face );
    }
    subject->setSurface( bothSurface );
    wlog::info( CLASS ) << "Successfully combine left and right surface!";

    return rc;
}

std::set< std::string > WLReaderExperiment::findLeadfieldTrials()
{
    wlog::debug( CLASS ) << "findLeadfieldTrials() called!";
    path path( m_PATH_EXPERIMENT );
    path /= m_FOLDER_RESULTS;
    path /= m_SUBJECT;

    std::set< std::string > trials;
    if( !exists( path ) )
    {
        wlog::error( CLASS ) << path.string() << " does not exist!";
        return trials;
    }

    std::string tmp, trial;
    size_t pos;
    directory_iterator end;
    for( directory_iterator it( path ); it != end; ++it )
    {
        if( is_regular_file( it->status() ) )
        {
            tmp = it->path().filename().string();
            if( tmp.find( m_LEADFIELD ) == 0 )
            {
                pos = tmp.find( m_SUBJECT );
                if( pos != std::string::npos )
                {
                    trial = tmp.substr( pos + m_SUBJECT.length(), 1 );
                    trials.insert( trial );
                }

            }
        }
    }

    wlog::debug( CLASS ) << "trials: " << trials.size();
    return trials;
}

bool WLReaderExperiment::readLeadFields( std::string surface, std::string bemName, std::string trial,
                LaBP::WLEMMSubject::SPtr subject )
{
    bool rc = true;
    rc &= readLeadField( surface, bemName, trial, m_EEG, subject );
    rc &= readLeadField( surface, bemName, trial, m_MEG, subject );
    return rc;
}

bool WLReaderExperiment::readLeadField( std::string surface, std::string bemName, std::string trial, std::string modality,
                LaBP::WLEMMSubject::SPtr subject )
{
    wlog::debug( CLASS ) << "readLeadField() called!";
    bool rc = true;

    bemName = bemName.substr( 0, bemName.find( m_VOL ) );

    path folder( m_PATH_EXPERIMENT );
    folder /= m_FOLDER_RESULTS;
    folder /= m_SUBJECT;

    path lhFile( folder );
    lhFile /= m_LEADFIELD + "_" + m_LH + "." + surface + "_" + bemName + "_" + m_SUBJECT + trial + "_" + modality + m_MAT;
    wlog::info( CLASS ) << "Read file: " << lhFile.string();
    if( !exists( lhFile ) )
    {
        wlog::error( CLASS ) << "File does not exist!";
        return false;
    }

    WLReaderMatMab lhReader( lhFile.string() );
    LaBP::MatrixSPtr lhMatrix;
    WLReaderMatMab::ReturnCode::Enum lhReturn = lhReader.read( lhMatrix );
    if( lhReturn == WLReaderMatMab::ReturnCode::SUCCESS )
    {
        wlog::info( CLASS ) << "Successfully read left leadfield!";
    }
    else
    {
        wlog::error( CLASS ) << "Could not load left leadfield!";
        return false;
    }

    path rhFile( folder );
    rhFile /= m_LEADFIELD + "_" + m_RH + "." + surface + "_" + bemName + "_" + m_SUBJECT + trial + "_" + modality + m_MAT;
    wlog::info( CLASS ) << "Read file: " << rhFile.string();
    if( !exists( rhFile ) )
    {
        wlog::error( CLASS ) << "File does not exist!";
        return false;
    }

    WLReaderMatMab rhReader( rhFile.string() );
    LaBP::MatrixSPtr rhMatrix;
    WLReaderMatMab::ReturnCode::Enum rhReturn = rhReader.read( rhMatrix );
    if( rhReturn == WLReaderMatMab::ReturnCode::SUCCESS )
    {
        wlog::info( CLASS ) << "Successfully read right leadfield!";
    }
    else
    {
        wlog::error( CLASS ) << "Could not load right leadfield!";
        return false;
    }

    // Combine lhMatrix and rhMatrix
    wlog::debug( CLASS ) << "lhMatrix " << lhMatrix->rows() << " x " << lhMatrix->cols();
    wlog::debug( CLASS ) << "rhMatrix " << rhMatrix->rows() << " x " << rhMatrix->cols();

    WAssert( lhMatrix->rows() == rhMatrix->rows(), "lh and rh matrices has different row size" );
    LaBP::MatrixSPtr leadfield( new LaBP::MatrixT( lhMatrix->rows(), lhMatrix->cols() + rhMatrix->cols() ) );

    for( LaBP::MatrixT::Index row = 0; row < leadfield->rows(); ++row )
    {
        for( LaBP::MatrixT::Index col = 0; col < lhMatrix->cols(); ++col )
        {
            ( *leadfield )( row, col ) = ( *lhMatrix )( row, col );
        }
        for( LaBP::MatrixT::Index col = 0; col < rhMatrix->cols(); ++col )
        {
            ( *leadfield )( row, col + lhMatrix->cols() ) = ( *rhMatrix )( row, col );
        }
    }

    wlog::debug( CLASS ) << "leadfield " << leadfield->rows() << " x " << leadfield->cols();

    if( m_MEG.compare( modality ) == 0 )
    {
        subject->setLeadfield( WEModalityType::MEG, leadfield );
    }
    else
        if( m_EEG.compare( modality ) == 0 )
        {
            subject->setLeadfield( WEModalityType::EEG, leadfield );
        }
        else
        {
            wlog::warn( CLASS ) << "Unknown modality. Leadfield is not stored!";
            return false;
        }

    return rc;
}
