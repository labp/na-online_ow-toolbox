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

#ifndef WWRITERFIFF_H_
#define WWRITERFIFF_H_

#include <string>

#include <boost/shared_ptr.hpp>
#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_info.h>
#include <fiff/fiff_stream.h>
#include <QFile>
#include <QList>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDMEG.h"
#include <core/dataHandler/exceptions/WDHIOFailure.h>
#include <core/dataHandler/io/WWriter.h>

class WWriterFiff: public WWriter
{
public:
    static const std::string CLASS;

    /**
     * Shared pointer abbreviation to a instance of this class.
     */
    typedef boost::shared_ptr< WWriterFiff > SPtr;

    /**
     * Shared pointer abbreviation to a const instance of this class.
     */
    typedef boost::shared_ptr< const WWriterFiff > ConstSPtr;

    WWriterFiff( std::string fname, bool pickEEG = true, bool pickMEG = true, bool pickStim = true, bool overwrite = false )
                    throw( WDHIOFailure );
    virtual ~WWriterFiff();

    bool open();
    bool close();

    bool write( WLEMMeasurement::ConstSPtr emm );

private:
    const bool m_pickEEG;
    const bool m_pickMEG;
    const bool m_pickStim;

    QFile* m_file;

    FIFFLIB::FiffStream::SPtr m_fiffStream;

    bool beginFiff( const WLEMMeasurement* const emm );

    bool setDigPoint( FIFFLIB::FiffInfo* const info, const WLEMMeasurement* const emm );

    bool writeData( const WLEMMeasurement* const emm );

    void setChannelInfo( FIFFLIB::FiffChInfo* const chInfo );
    void setChannelInfo( QList< FIFFLIB::FiffChInfo >* const chs, const WLEMMeasurement::EDataT* const stim );
    void setChannelInfo( QList< FIFFLIB::FiffChInfo >* const chs, const WLEMDEEG* const eeg );
    void setChannelInfo( QList< FIFFLIB::FiffChInfo >* const chs, const WLEMDMEG* const meg );
};

#endif  // WWRITERFIFF_H_
