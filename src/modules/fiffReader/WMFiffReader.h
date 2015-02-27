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

#ifndef WMFIFFREADER_H_
#define WMFIFFREADER_H_

#include <string>

#include <core/common/WCondition.h>
#include <core/common/WItemSelection.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WDataModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMSubject.h"
#include "core/io/WLReaderExperiment.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

/**
 * Reads a FIFF file and retrieves additional data, e.g. BEMs, surfaces or leadfields.
 *
 * \author pieloth
 * \ingroup io
 */
class WMFiffReader: public WDataModule
{
public:
    WMFiffReader();
    virtual ~WMFiffReader();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

    virtual std::vector< WDataModuleInputFilter::ConstSPtr > getInputFilter() const;

protected:
    virtual void handleInputChange();

    virtual void connectors();

    virtual void properties();

    virtual void moduleInit();

    virtual void moduleMain();

private:
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; //!<  Output connector for buffered input connectors.

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    WLEMMeasurement::SPtr m_emm;
    WLEMMSubject::SPtr m_subject;

    WPropTrigger m_trgSendEMM;
    void hdlTrgSendEMM();

    struct EFileStatus
    {
        enum Enum
        {
            NO_FILE, LOADING_FILE, FILE_ERROR, SUCCESS
        };

        static std::string name( Enum val );
    };
    EFileStatus::Enum m_fileStatus;
    WPropString m_propFileStatus;
    void updateFileStatus( EFileStatus::Enum status );

    bool m_reloadFiff;
    void hdlFiffFileChanged();
    bool readFiffFile( const std::string& fName );

    struct EDataStatus
    {
        enum Enum
        {
            NO_DATA, DATA_AVAILABLE, LOADING_DATA, DATA_ERROR, SUCCESS
        };

        static std::string name( Enum val );
    };
    EDataStatus::Enum m_dataStatus;
    WPropString m_propDataStatus;
    void updateDataStatus( EDataStatus::Enum status );

    WPropTrigger m_trgLoadData;
    void hdlTrgLoad();
    bool readData();

    bool retrieveAdditionalData( const std::string& fName );
    WLReaderExperiment::SPtr m_expReader;

    WPropString m_propSubject;

    WItemSelection::SPtr m_itmBemFiles;
    WPropSelection m_selBemFiles;

    WItemSelection::SPtr m_itmSurfaces;
    WPropSelection m_selSurfaces;

    WPropString m_propTrial;
};

inline void WMFiffReader::updateFileStatus( EFileStatus::Enum status )
{
    m_fileStatus = status;
    m_propFileStatus->set( EFileStatus::name( status ), true );
}

inline void WMFiffReader::updateDataStatus( EDataStatus::Enum status )
{
    m_dataStatus = status;
    m_propDataStatus->set( EDataStatus::name( status ), true );
}

#endif  // WMFIFFREADER_H_
