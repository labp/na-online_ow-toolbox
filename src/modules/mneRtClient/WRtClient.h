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

#ifndef WRTCLIENT_H_
#define WRTCLIENT_H_

#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <QtGlobal>

// MNE Library
#include <fiff/fiff_info.h>
#include <rtClient/rtcmdclient.h>
#include <rtClient/rtdataclient.h>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDMEG.h"

class WRtClient
{
public:
    typedef boost::shared_ptr< WRtClient > SPtr;
    typedef boost::shared_ptr< const WRtClient > ConstSPtr;

    static const std::string CLASS;

    WRtClient( const std::string& ip_address, const std::string& alias );
    virtual ~WRtClient();

    bool connect();

    bool isConnected();

    void disconnect();

    bool start();

    bool stop();

    bool isStreaming();

    int getConnectors( std::map< int, std::string >* const conMap );
    bool setConnector( int conId );

    bool setSimulationFile( std::string simFile );

    bool readData( WLEMMeasurement::SPtr emmIn );

private:
    bool m_isStreaming;
    bool m_isConnected;

    FIFFLIB::FiffInfo::SPtr m_fiffInfo;

    Eigen::RowVectorXi m_picksEeg;
    Eigen::RowVectorXi m_picksMeg;
    Eigen::RowVectorXi m_picksStim;

    bool readChannelPositionsFaces();
    boost::shared_ptr< std::vector< WPosition > > m_chPosEeg;
    boost::shared_ptr< std::vector< WVector3i > > m_facesEeg;
    boost::shared_ptr< std::vector< WPosition > > m_chPosMeg;

    bool readChannelNames();
    boost::shared_ptr< std::vector< std::string > > m_chNamesEeg;
    boost::shared_ptr< std::vector< std::string > > m_chNamesMeg;

    RTCLIENTLIB::RtCmdClient::SPtr m_rtCmdClient;
    RTCLIENTLIB::RtDataClient::SPtr m_rtDataClient;

    const std::string m_ipAddress;
    const std::string m_alias;
    qint32 m_clientId;

    WLEMDEEG::SPtr readEEG( const Eigen::MatrixXf& rawData );
    WLEMDMEG::SPtr readMEG( const Eigen::MatrixXf& rawData );
    boost::shared_ptr< WLEMMeasurement::EDataT > readEvents( const Eigen::MatrixXf& rawData );
    bool readEmd( WLEMData* const emd, const Eigen::RowVectorXi& picks, const Eigen::MatrixXf& rawData );
};

#endif  // WRTCLIENT_H_
