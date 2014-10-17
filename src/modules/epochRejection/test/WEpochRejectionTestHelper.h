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

#ifndef WEPOCHREJECTIONTESTHELPER_H_
#define WEPOCHREJECTIONTESTHELPER_H_

#include <algorithm>
#include <iostream>
#include <list>
#include <string>

#include "core/common/WLogger.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"

typedef boost::shared_ptr< Eigen::MatrixXd > MatrixSPtr;

class WEpochRejectionTestHelper
{
public:
    typedef boost::shared_ptr< WEpochRejectionTestHelper > SPtr;

    WEpochRejectionTestHelper()
    {
        WLogger::startup();

        CLASS = "WEpochRejectionTestHelper";

        THRESHOLD_EEG = 150e-6;
        THRESHOLD_EOG = 120e-6;
        THRESHOLD_MEG_GRAD = 200e-12;
        THRESHOLD_MEG_MAG = 5e-12;

        REJECTION_FAKTOR = 1.5;
    }

    /**
     * Method to create a MEG modality.
     */
    WLEMDMEG::SPtr createModality( size_t channels, size_t samples, size_t rejections, WLEMDMEG::SPtr emd )
    {
        if( channels % 3 != 0 )
            return emd;

        MatrixSPtr matrix_ptr( new Eigen::MatrixXd( channels, samples ) );
        Eigen::MatrixXd& matrix = *matrix_ptr;
        size_t channelno = 0;
        std::list< size_t > chanRejctIdx = getRecjetionIndex( channels, rejections ); // list to define in which channel has a rejection to be.

        for( size_t i = 0; i < channels; ++i ) // create the channels
        {
            ++channelno;

            size_t numOfRejections = std::find( chanRejctIdx.begin(), chanRejctIdx.end(), i ) == chanRejctIdx.end() ? 0 : 1;

            if( channelno % 3 == 0 ) // magnetometer
            {
                matrix.row( i ) = createRow( samples, ( THRESHOLD_MEG_MAG / 2 ), -( THRESHOLD_MEG_MAG / 2 ), numOfRejections );
            }
            else // gradiometer
            {
                matrix.row( i ) = createRow( samples, ( THRESHOLD_MEG_GRAD / 2 ), -( THRESHOLD_MEG_GRAD / 2 ), numOfRejections );
            }
        }

        // insert the data into the emd object.
        MatrixSPtr p( new Eigen::MatrixXd( matrix ) );
        emd->setData( p );

        return emd;
    }

    WLEMData::SPtr createModality( size_t channels, size_t samples, size_t rejections, double threshold, WLEMData::SPtr emd )
    {
        MatrixSPtr matrix_ptr( new Eigen::MatrixXd( channels, samples ) );
        Eigen::MatrixXd& matrix = *matrix_ptr;
        std::list< size_t > chanRejctIdx = getRecjetionIndex( channels, rejections ); // list to define in which channel has a rejection to be.

        for( size_t i = 0; i < channels; ++i ) // create the channels
        {
            size_t numOfRejections = std::find( chanRejctIdx.begin(), chanRejctIdx.end(), i ) == chanRejctIdx.end() ? 0 : 1;

            matrix.row( i ) = createRow( samples, ( threshold / 2 ), -( threshold / 2 ), numOfRejections );
        }

        // insert the data into the emd
        MatrixSPtr p( new Eigen::MatrixXd( matrix ) );
        emd->setData( p );

        return emd;
    }

private:
    std::string CLASS;

    double THRESHOLD_EEG;
    double THRESHOLD_EOG;
    double THRESHOLD_MEG_GRAD;
    double THRESHOLD_MEG_MAG;

    double REJECTION_FAKTOR;

    /**
     *
     * Creates a vector with [size] components within the value area of [min] - [max] including a number of rejections.
     *
     * \param size
     * \param max
     * \param min
     * \param rejections
     * \return
     */
    Eigen::VectorXd createRow( size_t size, double max, double min, size_t rejections )
    {
        Eigen::VectorXd row( size );

        std::list< size_t > rejectionIdx = getRecjetionIndex( size, rejections );

        for( size_t i = 0; i < size; ++i )
        {
            row( i ) = getRandD( max, min );

            if( rejectionIdx.size() == 0 )
                continue;

            if( i == rejectionIdx.front() )
            {
                size_t index = rejectionIdx.front();

                if( row( index ) < 0 )
                {
                    double x = min < 0 ? min : -min;
                    x *= REJECTION_FAKTOR;

                    row( index ) += x;
                }
                else
                {
                    double x = max < 0 ? -max : max;
                    x *= REJECTION_FAKTOR;

                    row( index ) += x;
                }

                rejectionIdx.pop_front();
            }
        }

        return row;
    }

    /**
     *
     * Returns a double value between two borders.
     *
     * \param min
     * \param max
     *
     * \return
     */
    double getRandD( double min, double max )
    {
        return ( max - min ) * ( ( double )rand() / ( double )RAND_MAX ) + min;
    }

    /**
     *
     * Returns a list of indices, which specifies, at which position a rejection has to be placed.
     *
     * \param size
     * \param rejections
     *
     * \return
     */
    std::list< size_t > getRecjetionIndex( size_t size, size_t rejections )
    {
        std::list< size_t > list;

        if( rejections == 0 )
            return list;

        while( list.size() < rejections )
        {
            size_t v = rand() % size;

            if( std::find( list.begin(), list.end(), v ) == list.end() )
                list.push_back( v );
        }

        list.sort();

        return list;
    }
};

#endif  // WEPOCHREJECTIONTESTHELPER_H_
