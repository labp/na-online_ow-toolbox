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

#ifndef WMEMMEASUREMENT_H
#define WMEMMEASUREMENT_H

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/common/WPropertyTypes.h>

#include "core/container/WLArrayList.h"
#include "core/container/WLList.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMSurface.h"
#include "core/data/WLEMMBemBoundary.h"
#include "core/data/WLDataTypes.h"

#include "core/module/WLModuleDrawable.h"

#include "core/io/WLReaderExperiment.h"

/**
 * This module implements several onscreen status displays. At the moment the main purpose
 * is the display of information from picking, i.e. what is picked.
 *
 * \authors kaehler, pieloth
 * \ingroup io
 */
class WMEmMeasurement: public WLModuleDrawable
{
public:
    /**
     * standard constructor
     */
    WMEmMeasurement();

    /**
     * destructor
     */
    virtual ~WMEmMeasurement();

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

    /**
     * Due to the prototype design pattern used to build modules, this method returns a new instance of this method. NOTE: it
     * should never be initialized or modified in some other way. A simple new instance is required.
     *
     * \return the prototype used to create every module in OpenWalnut.
     */
    virtual WModule::SPtr factory() const;

    /**
     * Get the icon for this module in XPM format.
     * \return The icon.
     */
    virtual const char** getXPMIcon() const;

protected:
    // ---------------------------------
    // Methods for WLEMMCommandProcessor
    // ---------------------------------
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );
    virtual bool processMisc( WLEMMCommand::SPtr labp );

    virtual void moduleInit();

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

private:
    //! a condition for the matrix selection
    WCondition::SPtr m_propCondition;

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
    WLEMMeasurement::SPtr m_fiffEmm;

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

    void setAdditionalInformation( WLEMMeasurement::SPtr emm );

    // ELC Settings //
    bool readElc( std::string fname );

    bool m_isElcLoaded;

    WPropFilename m_elcFile;

    WLArrayList< std::string >::SPtr m_elcLabels;

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

    WLEMMSurface::SPtr m_dipSurface;

    WPropString m_dipFileStatus;

    WPropInt m_dipPositionCount;

    WPropInt m_dipFacesCount;

    // VOL Settings //
    bool readVol( std::string fname );

    bool m_isVolLoaded;

    WPropFilename m_volFile;

    WLList< WLEMMBemBoundary::SPtr >::SPtr m_volBoundaries;

    WPropString m_volFileStatus;

    WPropInt m_volBoundaryCount;

    // Experiment loader //
    WPropGroup m_propGrpExperiment;
    WLEMMSubject::SPtr m_subject;
    bool m_isExpLoaded;

    void extractExpLoader( std::string fiffFile );
    WPropString m_expSubject;

    WItemSelection::SPtr m_expBemFiles;
    WPropSelection m_expBemFilesSelection;

    WItemSelection::SPtr m_expSurfaces;
    WPropSelection m_expSurfacesSelection;

    WPropString m_expTrial;
    // boost::shared_ptr< WItemSelection > m_expTrials;
    // WPropSelection m_expTrialsSelection;

    WPropTrigger m_expLoadTrigger;
    WPropString m_expLoadStatus;
    void handleExperimentLoadChanged();
    WLReaderExperiment::SPtr m_expReader;
};

#endif  // WMEMMEASUREMENT_H
