#ifndef LFINTERFACE_H
#define LFINTERFACE_H

#include "LFReturnCodes.h"
#include "LFData.h"

/**
 * Interface
 */
class LFInterface
{
public:
    /**
     * Reads the fiff - file and returns the LFFiffData - Object
     */
    static returncode_t fiffRead( LFData& output, const char* path );
    /**
     * Reads the fiff - file and returns the LFSubject - Object
     */
    static returncode_t fiffRead( LFSubject& output, const char* path );
    /**
     * Reads the fiff - file and returns the LFRawData - Object
     */
    static returncode_t fiffRead( LFRawData& output, const char* path );
};

#endif
