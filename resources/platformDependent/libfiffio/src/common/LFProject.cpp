#include "LFProject.h"

LFProject::LFProject() :
    m_Id( -1 )
{

}

void LFProject::Init()
{
    m_Id = -1;
    m_Name.clear();
    m_Comment.clear();
}

int32_t LFProject::GetId()
{
    return m_Id;
}

string& LFProject::GetName()
{
    return m_Name;
}

string& LFProject::GetComment()
{
    return m_Comment;
}

void LFProject::SetId( const int32_t src)
{
    m_Id=src;
}
