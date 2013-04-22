#ifndef LFMEASUREMENTINFO_H
#define LFMEASUREMENTINFO_H

#include <inttypes.h>
#include <vector>
using namespace std;

#include "LFArrayPtr.h"
#include "LFSubject.h"
#include "LFProject.h"
#include "LFHPIMeasurement.h"
#include "LFSSP.h"
#include "LFEvents.h"
#include "LFDataAcquisitionParameters.h"
#include "LFChannelInfo.h"
#include "LFHPISubsystem.h"
#include "LFIsotrak.h"

/**
 * Measurement-Info block (101)
 */
class LFMeasurementInfo
{
protected:
    LFSubject m_LFSubject;/**< Subject block (106) */
    LFProject m_LFProject;/**< Project block (111) */
    LFHPIMeasurement m_LFHPIMeasurement;/**< HPI Measurement block (108) */
    LFIsotrak m_LFIsotrak;/**< Isotrak block (107) */
    LFSSP m_LFSSP;/**< SSP block (313) */
    LFEvents m_LFEvents;/**< Events block (115) */
    LFDataAcquisitionParameters m_LFDataAcquisitionParameters;/**< Data Acquisition Parameters block (117) */
    int32_t m_NumberOfChannels;/**< Number of Channels, default == -1 (200) */
    float m_SamplingFrequency;/**< Sampling Frequency, Hz, default == -FLT_MAX (201) */
    float m_Lowpass;/**< Analog lowpass, Hz, default == -FLT_MAX (219) */
    float m_Highpass;/**< Analog highpass, Hz, default == -FLT_MAX (223) */
    int32_t m_DataPack;/**< How the raw data is packed, default == -1 (202) */
    float m_LineFreq;/**< Basic line interference frequency, Hz, default == -FLT_MAX (235) */
    int32_t m_GantryAngle;/**< Tilt angle of the dewar in degrees, default == 0 (282) */
    LFArrayPtr<LFChannelInfo> m_LFChannelInfo;/**< Channel descriptor (203) */
    vector< int32_t > m_BadChannels;/**< List of bad channels (220) */
    LFHPISubsystem m_LFHPISubsystem;/**< HPI Subsystem block (121) */
public:
    /**
     * Constructor
     */
    LFMeasurementInfo();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns Subject block
     */
    LFSubject& GetLFSubject();
    /**
     * Returns Project block
     */
    LFProject& GetLFProject();
    /**
     * Returns HPI Measurement block
     */
    LFHPIMeasurement& GetLFHPIMeasurement();
    /**
     * Returns the Isotrak block
     */
    LFIsotrak& GetLFIsotrak();
    /**
     * Returns SSP block
     */
    LFSSP& GetLFSSP();
    /**
     * Returns Events block
     */
    LFEvents& GetLFEvents();
    /**
     * Returns Data Acquisition Parameters block
     */
    LFDataAcquisitionParameters& GetLFDataAcquisitionParameters();
    /**
     * Returns the Number of Channels
     */
    int32_t GetNumberOfChannels();
    /**
     * Returns the Sampling Frequency
     */
    float GetSamplingFrequency();
    /**
     * Returns the analog lowpass
     */
    float GetLowpass();
    /**
     * Returns the analog highpass
     */
    float GetHighpass();
    /**
     * Returns the information about how the raw data is packed
     */
    int32_t GetDataPack();
    /**
     * Returns the basic line interference frequency
     */
    float GetLineFreq();
    /**
     * Returns the tilt angle of the dewar in degrees
     */
    int32_t GetGantryAngle();
    /**
     * Returns the channel descriptors
     */
    LFArrayPtr<LFChannelInfo>& GetLFChannelInfo();
    /**
     * Returns the list of bad channels
     */
    vector< int32_t >& GetBadChannels();
    /**
     * Returns the HPI Subsystem block
     */
    LFHPISubsystem& GetLFHPISubsystem();
    /**
     * Sets the Number of Channels
     */
    void SetNumberOfChannels( const int32_t src );
    /**
     * Sets the Sampling Frequency
     */
    void SetSamplingFrequency( const float src );
    /**
     * Sets the analog lowpass
     */
    void SetLowpass( const float src );
    /**
     * Sets the analog highpass
     */
    void SetHighpass( const float src );
    /**
     * Sets the information about how the raw data is packed
     */
    void SetDataPack( const int32_t src );
    /**
     * Sets the basic line interference frequency
     */
    void SetLineFreq( const float src );
    /**
     * Sets the tilt angle of the dewar in degrees
     */
    void SetGantryAngle( const int32_t src );
};

#endif
