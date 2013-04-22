#ifndef LFPROJECT_H
#define LFPROJECT_H

#include <inttypes.h>
#include <string>
using namespace std;

/**
 * Project block (111), Information about the project under which the data were acquired
 */
class LFProject
{
protected:
    int32_t m_Id;/**< Project ID, default == -1 (500) */
    string m_Name;/**< Project name (501) */
    string m_Comment;/**< Project description (504) */
public:
    /**
     * Constructor
     */
    LFProject();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns Project ID
     */
    int32_t GetId();
    /**
     * Returns Project name
     */
    string& GetName();
    /**
     * Returns Project description
     */
    string& GetComment();
    /**
     * Sets Project ID
     */
    void SetId( const int32_t src);
};
#endif
