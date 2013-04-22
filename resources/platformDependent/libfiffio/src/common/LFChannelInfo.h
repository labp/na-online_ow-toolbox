#ifndef LFCHANNELINFO_H
#define LFCHANNELINFO_H

#include <inttypes.h>
using namespace std;

#include "LFUnits.h"
#include "LFMultipliers.h"

/**
 * ch_info_t structure, contains information about one channel
 */
class LFChannelInfo
{
protected:
    int32_t m_ScanNo;/**< Position of this channel in scanning order, default == -1 */
    int32_t m_LogNo;/**< Logical channel number. These must be unique within channels of the same kind, default ==-1 */
    int32_t m_Kind;/**< Kind of the channel described (MEG, EEG, EOG, etc.), default == -1 */
    float m_Range;/**< The raw data values must be multiplied by this to get into volts at the electronics output, default == -FLT_MAX */
    float m_Cal;/**< Calibration of the channel. If the raw data values are multiplied by range*cal, the result is in units given by unit and unit_mul. */
    int32_t m_CoilType;/**< Kind of MEG coil or kind of EEG channel. */
    float m_R0[3];/**< Coil coordinate system origin. For EEG electrodes, this is the location of the electrode. */
    float m_Ex[3];/**< Coil coordinate system unit vector e x. For EEG electrodes, this specifies the location of a reference electrode. Set to (0,0,0) for no reference. */
    float m_Ey[3];/**< Coil coordinate system unit vector e y . This is ignored for EEG electrodes. */
    float m_Ez[3];/**< Coil coordinate system unit vector e z . This is ignored for EEG electrodes. */
    fiffunits_t m_Unit;/**< The real-world unit of Measure */
    fiffmultipliers_t m_UnitMul;/**< The unit multiplier. The result given by range*cal*data is in units unit*10^unit_mul. */
public:
    /**
     * Constructor
     */
    LFChannelInfo();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns position of this channel in scanning order
     */
    int32_t GetScanNo();
    /**
     * Returns logical channel number
     */
    int32_t GetLogNo();
    /**
     * Returns kind of the channel described (MEG, EEG, EOG, etc.)
     */
    int32_t GetKind();
    /**
     * Returns the multiplication coefficient
     */
    float GetRange();
    /**
     * Returns the calibration of the channel.
     */
    float GetCal();
    /**
     * Returns the kind of MEG coil or kind of EEG channel.
     */
    int32_t GetCoilType();
    /**
     * Returns the coil coordinate system origin.
     */
    float* GetR0();
    /**
     * Returns the coil coordinate system unit vector e x
     */
    float* GetEx();
    /**
     * Returns the coil coordinate system unit vector e y
     */
    float* GetEy();
    /**
     * Returns the coil coordinate system unit vector e z
     */
    float* GetEz();
    /**
     * Returns the real-world unit of Measure
     */
    fiffunits_t GetUnit();
    /**
     * Returns the unit multiplier.
     */
    fiffmultipliers_t GetUnitMul();
    /**
     * Sets position of this channel in scanning order
     */
    void SetScanNo( const int32_t src );
    /**
     * Sets logical channel number
     */
    void SetLogNo( const int32_t src );
    /**
     * Sets kind of the channel described (MEG, EEG, EOG, etc.)
     */
    void SetKind( const int32_t src );
    /**
     * Sets the multiplication coefficient
     */
    void SetRange( const float src );
    /**
     * Sets the calibration of the channel.
     */
    void SetCal( const float src );
    /**
     * Sets the kind of MEG coil or kind of EEG channel.
     */
    void SetCoilType( const int32_t src );
    /**
     * Sets the coil coordinate system origin.
     */
    void SetR0( const float* src );
    /**
     * Sets the coil coordinate system unit vector e x
     */
    void SetEx( const float* src );
    /**
     * Sets the coil coordinate system unit vector e y
     */
    void SetEy( const float* src );
    /**
     * Sets the coil coordinate system unit vector e z
     */
    void SetEz( const float* src );
    /**
     * Sets the real-world unit of Measure
     */
    void SetUnit( const fiffunits_t src );
    /**
     * Sets the unit multiplier.
     */
    void SetUnitMul( const fiffmultipliers_t src );
};
#endif
