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

#ifndef WHEADSIGNALEXTRACTION_H_
#define WHEADSIGNALEXTRACTION_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include "core/data/WLDataTypes.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/emd/WLEMDHPI.h"

#include "../firFilter/WFIRFilter.h"

/**
 * Reconstructs the amplitudes of each HPI coil on each MEG sensor.
 *
 * Using approaches from:
 * D.1 HPI-Signals, p. 73, "MaxFilter - User's Guide"; Elekta Neuromag, 2009
 * "Detecting and Correcting for Head Movements in Neuromagnetic Measurements"; Uutela et al., 2001
 *
 * \author pieloth
 */
class WHPISignalExtraction
{
public:
    static const std::string CLASS;

    typedef boost::shared_ptr< WHPISignalExtraction > SPtr;

    typedef boost::shared_ptr< const WHPISignalExtraction > ConstSPtr;

    WHPISignalExtraction();
    virtual ~WHPISignalExtraction();

    /**
     * Gets the windows size.
     *
     * \return Windows size in ms.
     */
    WLTimeT getWindowsSize() const;

    /**
     * Sets the windows size. Requested size could be change if it is to small!
     *
     * \param winSize Windows size in ms.
     * \return set windows size in ms
     */
    WLTimeT setWindowsSize( WLTimeT winSize );

    /**
     * Gets the step size.
     *
     * \return Step size in ms.
     */
    WLTimeT getStepSize() const;

    /**
     * Gets the sampling frequency.
     *
     * \return Sampling frequency in Hz.
     */
    WLFreqT getSamplingFrequency() const;

    /**
     * Sets the sampling frequency.
     *
     * \param sfreq Sampling frequency in Hz.
     */
    void setSamplingFrequency( WLFreqT sfreq );

    /**
     * Sets the step size.
     *
     * \param stepSize Step size in ms.
     */
    void setStepSize( WLTimeT stepSize );

    /**
     * Adds a frequency of a HPI coil. Transforms the frequency into angular frequency.
     *
     * \param freq Frequenz in Hz.
     */
    void addFrequency( WLFreqT freq );

    /**
     * Gets the frequencies.
     *
     * \return The frenquencies is Hz.
     */
    std::vector< WLFreqT > getFrequencies() const;

    /**
     * Clears the frequencies.
     *
     * \return Number of frequencies removed.
     */
    size_t clearFrequencies();

    /**
     * Pre-computes A, A^T and A^T*A matrix.
     *
     * \return true, if successful.
     */
    bool prepare();

    /**
     * Reconstructs the amplitudes on the MEG sensors.
     *
     * \param hpiOut Contains the output data, if successful
     * \param megIn MEG input data
     * \return true, if successful
     */
    bool reconstructAmplitudes( WLEMDHPI::SPtr& hpiOut, WLEMDMEG::ConstSPtr megIn );

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
     * \param hpiOut Sample vector to store reconstructed amplitudes.
     * \param megIn MEG data to read from.
     * \param start Start sample for the windows.
     * \param samples Samples to read/windows size.
     */
    void reconstructWindows( WLEMData::SampleT* const hpiOut, const WLEMData::DataT& megIn, MatrixT::Index start,
                    MatrixT::Index samples );

    WFIRFilter::SPtr m_firFilter;

    bool m_isPrepared; /**< Indicates if the algorithm is prepared. */
    WLTimeT m_windowsSize; /**< Windows size in seconds. */
    WLTimeT m_stepSize; /**< Step size in seconds. */
    WLFreqT m_sampFreq; /**< Sampling frequency. */

    std::vector< WLFreqT > m_angFrequencies; /**< Angular frequencies in Hz. */

    MatrixT m_at; /**< Transpose of A. Pre-calculated by prepare(). */
    MatrixT m_ata; /**< A^2*A. Pre-calculated by prepare(). */
};

inline std::ostream& operator<<( std::ostream &strm, const WHPISignalExtraction& obj )
{
    strm << WHPISignalExtraction::CLASS << ": ";
    strm << "winSize=" << obj.getWindowsSize() << ", ";
    strm << "stepSize=" << obj.getStepSize() << ", ";
    strm << "sFreq=" << obj.getSamplingFrequency() << ", ";
    strm << "hpiFreqs=" << obj.getFrequencies().size();
    return strm;
}

#endif  // WHEADSIGNALEXTRACTION_H_
