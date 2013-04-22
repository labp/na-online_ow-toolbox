#ifndef LFSUBJECT_H
#define LFSUBJECT_H

#include <inttypes.h>
#include<string>
using namespace std;

/**
 * Subject block (106)
 */
class LFSubject
{
public:
    enum sex_t/**< Patient's sex TODO(Evfimevskiy): die Werte sind in dem Standard nicht definiert */
    {
        sex_unknown=-1
        ,sex_m=0
        ,sex_f=1
    };
    enum hand_t/**< Right- or left-handed patient TODO(Evfimevskiy): die Werte sind in dem Standard nicht definiert */
    {
        hand_unknown=-1
        ,hand_right=0
        ,hand_left=1
    };
protected:
    string m_HIS_ID;/**< HIS ID (410) ID used in the Hospital Information System */
    string m_LastName;/**< Last Name (403) */
    string m_FirstName;/**< First Name (401) */
    string m_MiddleName;/**< Middle Name (402) */
    int32_t m_Birthday;/**< Birthdate (404) */
    sex_t m_Sex;/**< Patient's sex, default == sex_unknown (405) */
    hand_t m_Hand;/**< Right- or left-handed patient, default == hand_unknown (406) */
    float m_Weight;/**< Body weight, kg, default == -FLT_MAX (407) */
    float m_Height;/**< Body height, m, default == -FLT_MAX (408) */
    string m_Comment;/**< Comment (409) */
public:
    /**
     * Constructor
     */
    LFSubject();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the HIS ID
     */
    string& GetHIS_ID();
    /**
     * Returns the Last Name
     */
    string& GetLastName();
    /**
     * Returns the First Name
     */
    string& GetFirstName();
    /**
     * Returns the Middle Name
     */
    string& GetMiddleName();
    /**
     * Returns the Birthdate
     */
    int32_t GetBirthday();
    /**
     * Returns patient's sex
     */
    sex_t GetSex();
    /**
     * Right- or left-handed patient
     */
    hand_t GetHand();
    /**
     * Returns the body weight
     */
    float GetWeight();
    /**
     * Returns the body height
     */
    float GetHeight();
    /**
     * Returns the comment
     */
    string& GetComment();
    /**
     * Sets the HIS ID
     */
    void SetHIS_ID( const char* src );
    /**
     * Sets the den Last Name
     */
    void SetLastName( const char* src );
    /**
     * Sets the First Name
     */
    void SetFirstName( const char* src );
    /**
     * Sets the Middle Name
     */
    void SetMiddleName( const char* src );
    /**
     * Sets the Birthdate
     */
    void SetBirthday( const int32_t src );
    /**
     * Sets patient's sex
     */
    void SetSex( const sex_t src );
    /**
     * Sets right- or left-handed patient
     */
    void SetHand( const hand_t src );
    /**
     * Sets the body weight
     */
    void SetWeight( const float src );
    /**
     * Sets the body height
     */
    void SetHeight( const float src );
    /**
     * Sets the Comment
     */
    void SetComment( const char* src );
};

#endif
