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

#ifndef WMEMMMEASUREMENT_H
#define WMEMMMEASUREMENT_H

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/common/math/linearAlgebra/WVectorFixed.h"
#include "core/common/WPropertyTypes.h"
#include "core/dataHandler/WDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMSubject.h"
#include "core/dataHandler/WDataSetEMMSurface.h"
#include "core/dataHandler/WDataSetEMMBemBoundary.h"

#include "core/kernel/WModuleEMMView.h"
// TODO use OW class
#include "core/kernel/WLModuleOutputDataCollectionable.h"

#include "algorithms/WRegistration.h"
#include "algorithms/WRegistrationICP.h"
#include "algorithms/WRegistrationNaive.h"

#include "reader/WReaderExperiment.h"

/**
 * This module implements several onscreen status displays. At the moment the main purpose
 * is the display of information from picking, i.e. what is picked.
 * \ingroup modules
 */
class WMemmMeasurement: public LaBP::WModuleEMMView
{
public:
    /**
     * standard constructor
     */
    WMemmMeasurement();

    /**
     * destructor
     */
    virtual ~WMemmMeasurement();

    /**
     * Returns the name of this module.
     * \return the module's name.
     */
    virtual const std::string getName() const;

    /**
     * Returns a description of this module.
     * \return description of module.
     */
    virtual const std::string getDescription() const;

protected:
    virtual void initModule();

    /**
     * Entry point after loading the module. Runs in separate thread.
     */
    virtual void moduleMain();

    /**
     * Initialize the connectors this module is using.
     */
    virtual void connectors();

    /**
     * Initialize the properties for this module.
     */
    virtual void properties();

    /**
     * Due to the prototype design pattern used to build modules, this method returns a new instance of this method. NOTE: it
     * should never be initialized or modified in some other way. A simple new instance is required.
     *
     * \return the prototype used to create every module in OpenWalnut.
     */
    virtual boost::shared_ptr< WModule > factory() const;

    /**
     * Get the icon for this module in XPM format.
     * \return The icon.
     */
    virtual const char** getXPMIcon() const;

private:
    //! a condition for the matrix selection
    boost::shared_ptr< WCondition > m_propCondition;

    /**
     * The only output of this data module. TODO use OW class
     */
    boost::shared_ptr< LaBP::WLModuleOutputDataCollectionable< LaBP::WDataSetEMM > > m_output;

    // FIFF file //
    void streamData();

    bool readFiff( std::string fname );

    bool m_isFiffLoaded;

    WPropInt m_fiffStreamBlockSize;

    /**
     * start streaming fiff file
     */
    WPropTrigger m_streamFiffTrigger;

    /**
     * pointer to data out of read fiff file
     */
    boost::shared_ptr< LaBP::WDataSetEMM > m_fiffEmm;

    WPropString m_fiffFileStatus;

    /**
     * Exported MatLAB File with coefficients
     */
    WPropFilename m_fiffFile;

    /**
     * Group for cutting fiff-file for stream
     */
    WPropGroup m_propGrpFiffStreaming;

    // Data generation //
    void generateData();

    WPropInt m_generationBlockSize;

    /**
     * Duration to generate Data in seconds
     */
    WPropInt m_generationNrChans;

    /**
     * Duration to generate Data in seconds
     */
    WPropInt m_generationDuration;

    /**
     * Frequency to generate Data in Hz
     */
    WPropInt m_generationFreq;

    /**
     * Group for data generation parameter
     */
    WPropGroup m_propGrpDataGeneration;

    /**
     * start generating Data
     */
    WPropTrigger m_genDataTrigger;

    /**
     * stop generating data
     */
    WPropTrigger m_genDataTriggerEnd;

    // Additional Settings //
    WPropGroup m_propGrpExtra;

    void setAdditionalInformation( boost::shared_ptr< LaBP::WDataSetEMM > emm );

    // ELC Settings //
    bool readElc( std::string fname );

    bool m_isElcLoaded;

    WPropFilename m_elcFile;

    boost::shared_ptr< std::vector< std::string > > m_elcLabels;

    boost::shared_ptr< std::vector< WPosition > > m_elcPositions3d;

    boost::shared_ptr< std::vector< WVector3i > > m_elcFaces;

    WPropString m_elcFileStatus;

    WPropInt m_elcChanLabelCount;

    WPropInt m_elcChanPositionCount;

    WPropInt m_elcFacesCount;

    // DIP Settings //
    bool readDip( std::string fname );

    bool m_isDipLoaded;

    WPropFilename m_dipFile;

    boost::shared_ptr< LaBP::WDataSetEMMSurface > m_dipSurface;

    WPropString m_dipFileStatus;

    WPropInt m_dipPositionCount;

    WPropInt m_dipFacesCount;

    // VOL Settings //
    bool readVol( std::string fname );

    bool m_isVolLoaded;

    WPropFilename m_volFile;

    boost::shared_ptr< std::vector< boost::shared_ptr< LaBP::WDataSetEMMBemBoundary > > > m_volBoundaries;

    WPropString m_volFileStatus;

    WPropInt m_volBoundaryCount;

    // Registration settings //
    WPropGroup m_propGrpRegistration;

    WRegistration::MatrixTransformation m_regTransformation;

    WRegistrationNaive m_regNaive;
    WRegistrationICP m_regICP;
    WPropTrigger m_regAlignTrigger;
    WPropDouble m_regError;

    void align();

    // Experiment loader //
    WPropGroup m_propGrpExperiment;
    LaBP::WDataSetEMMSubject::SPtr m_subject;
    bool m_isExpLoaded;

    void extractExpLoader( std::string fiffFile );
    WPropString m_expSubject;

    boost::shared_ptr< WItemSelection > m_expBemFiles;
    WPropSelection m_expBemFilesSelection;

    boost::shared_ptr< WItemSelection > m_expSurfaces;
    WPropSelection m_expSurfacesSelection;

    WPropString m_expTrial;
    // boost::shared_ptr< WItemSelection > m_expTrials;
    // WPropSelection m_expTrialsSelection;

    WPropTrigger m_expLoadTrigger;
    WPropString m_expLoadStatus;
    void handleExperimentLoadChanged();
    WReaderExperiment::SPtr m_expReader;

    // File status string //

    static const std::string NO_DATA_LOADED;

    static const std::string LOADING_DATA;

    static const std::string DATA_LOADED;

    static const std::string DATA_ERROR;

    static const std::string NO_FILE_LOADED;

    static const std::string LOADING_FILE;

    static const std::string FILE_LOADED;

    static const std::string FILE_ERROR;

};

#endif  // WMEMMMEASUREMENT_H
