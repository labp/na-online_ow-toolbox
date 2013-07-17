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

#include "core/common/WLogger.h"

#include "WEpochRejection.h"

const std::string WEpochRejection::CLASS = "WEpochRejection";

/**
 * Constructor
 */
WEpochRejection::WEpochRejection()
{
    initRejection();
}

/**
 * Destructor
 */
WEpochRejection::~WEpochRejection()
{

}

/**
 * Method to reset the thresholds.
 */
void WEpochRejection::initRejection()
{
   m_eegLevel = 0;
   m_eogLevel = 0;
   m_megGrad = 0;
   m_megMag = 0;
   m_rejCount = 0;
}

/**
 * Method to initialize the thresholds
 */
void WEpochRejection::setLevels(double eegLevel, double eogLevel, double megGrad, double megMag)
{
    m_eegLevel = eegLevel;
    m_eogLevel = eogLevel;
    m_megGrad = megGrad;
    m_megMag = megMag;
}

/**
 * Method to process the rejection on the data.
 */
bool WEpochRejection::getRejection(const LaBP::WLDataSetEMM::SPtr emm)
{
    wlog::debug( CLASS ) << "starting rejection";

    m_rejCount = 0;

    LaBP::WLEMD::SPtr modality;
    size_t channels;
    size_t samples;
    size_t rejections = 0;
    double levelValue = 0;

    for(size_t mod = 0; mod < emm->getModalityCount(); mod++) // for all modalities
    {
        // get modality
        modality = emm->getModality( mod );

        // if wrong modality, next
        if(! validModality(modality->getModalityType()))
        {
            wlog::debug( CLASS ) << "invalid modality";
            continue;
        }

        channels = modality->getData().size(); // get number of channels
        samples = modality->getData().front().size(); // get number of samples

        size_t channelCount = 0;

        for( size_t chan = 0; chan < channels; chan++) // for all channels
        {
            double min = 0, max = 0; // vars for min/max peak

            channelCount++; // number of the current channel

            // definition of the threshold to use by the modality
            switch(modality->getModalityType())
            {
                case LaBP::WEModalityType::EEG:
                    levelValue = m_eegLevel;
                    break;
                case LaBP::WEModalityType::EOG:
                    levelValue = m_eogLevel;
                    break;
                case LaBP::WEModalityType::MEG:
                    if( (channelCount % 3) == 0) // magnetometer
                        levelValue = m_megMag;
                    else // gradiometer
                        levelValue = m_megGrad;
                    break;
                default:; // skip the channel
            }

            for( size_t samp = 0; samp < samples; samp++ ) // for all samples of the channel
            {
                double sample = modality->getData()[chan][samp]; // store the sample value

                if( max == 0)
                    max = sample;

                if( min == 0)
                    min = sample;

                if(sample > max) // define maximum
                    max = sample;

                if(sample < min) // define minimum
                    min = sample;
            }

            double diff = 0;

            // calculate the difference between min and max peek for the channel
            if(max >= 0)
            {
                diff = max - min;
            }
            else
            {
                diff = min - max;
                diff *= (-1); // positive difference
            }

            // compare the difference with the given level value
            if(diff > levelValue)
            {
                rejections++; // counts the rejected for each modality
            }
        }

        // if least one channel has to reject, reject the whole input
        if(rejections > 0)
            m_rejCount++;
    }

    return m_rejCount > 0;
}

/**
 * Returns the number of rejections.
 */
size_t WEpochRejection::getCount()
{
    return m_rejCount;
}

/**
 * Method to separate valid modalities from invalid modalities.
 * It returns false, if the modality has to skip else true.
 */
bool WEpochRejection::validModality(LaBP::WEModalityType::Enum modalityType)
{
   bool rc = false;

   switch(modalityType)
   {
       case LaBP::WEModalityType::EEG:
           rc = true;
           break;
       case LaBP::WEModalityType::EOG:
           rc = true;
           break;
       case LaBP::WEModalityType::MEG:
           rc = true;
           break;
       default:;
   }

   return rc;
}
