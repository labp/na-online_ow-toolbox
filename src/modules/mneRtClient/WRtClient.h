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

#include <list>
#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <QtGlobal>

// MNE Library
#include <fiff/fiff_info.h>
#include <fiff/fiff_types.h>
#include <rtClient/rtcmdclient.h>
#include <rtClient/rtdataclient.h>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLList.h"
#include "core/data/WLDataTypes.h"
#include "core/data/WLDigPoint.h"
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

    bool isConnected() const;

    void disconnect();

    bool start();

    bool stop();

    bool isStreaming() const;

    int getConnectors( std::map< int, std::string >* const conMap );
    bool setConnector( int conId );

    bool setSimulationFile( std::string simFile );

    void setBlockSize(int blockSize);
    int getBlockSize() const;

    bool readData( WLEMMeasurement::SPtr& emmIn );

    bool isScalingApplied() const;
    void setScaling( bool applyScaling );

    // TODO(pieloth): Workaround for #227/#228
    bool setDigPointsAndEEG( const std::list< WLDigPoint >& digPoints );

private:
    typedef std::vector< std::string > ChannelNamesT;
    typedef boost::shared_ptr< ChannelNamesT > ChannelNamesSPtr;

    typedef std::vector< WPosition > ChannelsPositionsT;
    typedef boost::shared_ptr< ChannelsPositionsT > ChannelsPositionsSPtr;

    typedef std::vector< WVector3i > FacesT;
    typedef boost::shared_ptr< FacesT > FacesSPtr;

    bool m_isStreaming;
    bool m_isConnected;

    std::map< int, std::string > m_conMap;
    int m_conSelected;

    FIFFLIB::FiffInfo::SPtr m_fiffInfo;

    WLEMMeasurement::ConstSPtr m_emmPrototype;
    WLEMDEEG::ConstSPtr m_eegPrototype;
    WLEMDMEG::ConstSPtr m_megPrototype;

    ChannelsPositionsSPtr m_chPosEeg;
    FacesSPtr m_facesEeg;

    Eigen::RowVectorXi m_picksEeg;
    Eigen::RowVectorXi m_picksMeg;
    Eigen::RowVectorXi m_picksStim;

    std::vector<float> m_scaleFactors;
    bool m_applyScaling;

    bool prepareStreaming();
    bool preparePrototype( WLEMMeasurement* const emm );
    bool preparePrototype( WLEMData* const emd, const Eigen::RowVectorXi& picks );
    bool preparePrototype( WLEMDEEG* const emd, const Eigen::RowVectorXi& picks );
    bool preparePrototype( WLEMDMEG* const emd, const Eigen::RowVectorXi& picks );

    bool readChannelNames( WLEMData* const emd, const Eigen::RowVectorXi& picks );
    bool readChannelPositions( WLEMData* const emd, const Eigen::RowVectorXi& picks );
    bool readChannelFaces( WLEMData* const emd, const Eigen::RowVectorXi& picks );

    RTCLIENTLIB::RtCmdClient::SPtr m_rtCmdClient;
    RTCLIENTLIB::RtDataClient::SPtr m_rtDataClient;

    const std::string m_ipAddress;
    const std::string m_alias;
    qint32 m_clientId;

    int m_blockSize;

    WLList< WLDigPoint >::SPtr m_digPoints;

    bool readEmd( WLEMData* const emd, const Eigen::RowVectorXi& picks, const Eigen::MatrixXf& rawData );
    WLEMDEEG::SPtr readEEG( const Eigen::MatrixXf& rawData );
    WLEMDMEG::SPtr readMEG( const Eigen::MatrixXf& rawData );
    boost::shared_ptr< WLEMMeasurement::EDataT > readEvents( const Eigen::MatrixXf& rawData );
};

#endif  // WRTCLIENT_H_
