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

#ifndef WHEADPOSITIONESTIMATION_H_
#define WHEADPOSITIONESTIMATION_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include "core/data/WLDataTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDMEG.h"

/**
 * Reconstructs the amplitudes of each HPI coil on each MEG sensor.
 * Using approach from: D.1 HPI-Signals, p. 73, "MaxFilter - User's Guide"; Elekta Neuromag, 2009
 *
 * \author pieloth
 */
class WHPIAmplitudeReconstruction
{
public:
    static const std::string CLASS;

    typedef boost::shared_ptr< WHPIAmplitudeReconstruction > SPtr;

    typedef boost::shared_ptr< const WHPIAmplitudeReconstruction > ConstSPtr;

    WHPIAmplitudeReconstruction();
    virtual ~WHPIAmplitudeReconstruction();

    /**
     * Gets the windows size.
     *
     * @return Windows size in ms.
     */
    WLTimeT getWindowsSize() const;

    /**
     * Sets the windows size.
     *
     * @param winSize Windows size in ms.
     */
    void setWindowsSize( WLTimeT winSize );

    /**
     * Gets the step size.
     *
     * @return Step size in ms.
     */
    WLTimeT getStepSize() const;

    /**
     * Gets the sampling frequency.
     *
     * @return Sampling frequency in Hz.
     */
    WLFreqT getSamplingFrequency() const;

    /**
     * Sets the sampling frequency.
     *
     * @param sfreq Sampling frequency in Hz.
     */
    void setSamplingFrequency( WLFreqT sfreq );

    /**
     * Sets the step size.
     *
     * @param stepSize Step size in ms.
     */
    void setStepSize( WLTimeT stepSize );

    /**
     * Adds a frequency of a HPI coil. Transforms the frequency into angular frequency.
     *
     * @param freq Frequenz in Hz.
     */
    void addFrequency( WLFreqT freq );

    /**
     * Gets the frequencies.
     *
     * @return The frenquencies is Hz.
     */
    std::vector< WLFreqT > getFrequencies() const;

    /**
     * Clears the frequencies.
     *
     * @return Number of frequencies removed.
     */
    size_t clearFrequencies();

    /**
     * Pre-computes A, A^T and A^T*A matrix.
     *
     * @return true, if successful.
     */
    bool prepare();

    /**
     * Reconstructs the amplitudes on the MEG sensors.
     * TODO (pieloth): Do we need a highpass filter for input data?
     *
     * @param hpiOut Contains the output data, if successful
     * @param megIn MEG input data
     * @return true, if successful
     */
    bool reconstructAmplitudes( WLEMData::DataSPtr hpiOut, WLEMDMEG::ConstSPtr megIn );

    /**
     * Resets the algorithm.
     */
    void reset();

private:
    typedef WLMatrix::MatrixT MatrixT;
    typedef WLVector::VectorT VectorT;

    /**
     * Reconstructs the amplitudes for one windows.
     *
     * @param hpiOut Sample vector to store reconstructed amplitudes.
     * @param megIn MEG data to read from.
     * @param start Start sample for the windows.
     * @param samples Samples to read/windows size.
     */
    void reconstructWindows( WLEMData::SampleT* const hpiOut, const WLEMData::DataT& megIn, MatrixT::Index start,
                    MatrixT::Index samples );

    WLEMDMEG::ConstSPtr m_lastMeg; /**< Stores the previous MEG data block. */

    bool m_isPrepared; /**< Indicates if the algorithm is prepared. */
    WLTimeT m_windowsSize; /**< Windows size in seconds. */
    WLTimeT m_stepSize; /**< Step size in seconds. */
    WLFreqT m_sampFreq; /**< Sampling frequency. */

    std::vector< WLFreqT > m_angFrequencies; /**< Angular frequencies in Hz. */

    MatrixT m_at; /**< Transpose of A. Pre-calculated by prepare(). */
    MatrixT m_ata; /**< A^2*A. Pre-calculated by prepare(). */
};

#endif  // WHEADPOSITIONESTIMATION_H_
