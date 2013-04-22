#ifndef LFSSP_H
#define LFSSP_H

#include <inttypes.h>

#include "LFArrayPtr.h"
#include "LFProjectionItem.h"

/**
 * Signal space projections (SSP) Block (313)
 */
class LFSSP
{
protected:
    int32_t m_NumberOfChannels;/**< Number Of Channels, default == -1 (200) */
    LFArrayPtr<LFProjectionItem> m_LFProjectionItem;/**< Projection Item block (314) */
public:
    /**
     * Constructor
     */
    LFSSP();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the Number Of Channels
     */
    int32_t GetNumberOfChannels();
    /**
     * Returns the Projection Item blocks
     */
    LFArrayPtr<LFProjectionItem>& GetLFProjectionItem();
    /**
     * Sets the Number Of Channels
     */
    void SetNumberOfChannels( const int32_t src );
};

#endif
