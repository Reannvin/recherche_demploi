#include "../include/preprocess.h"

class Solution {
public:
    string exp(string s, int l, int r){
        while(l>0 && r<s.size() && s[l] == s[r]){
            l -- ;
            r ++ ;
        }
        return s.substr(l+1,r-l-1);
    }

    string longestPalindrome(string s){
        string res = "";
        for(int i = 0; i < s.size();i++){
            string s1 = exp(s,i,i);
            string s2 = exp(s,i,i+1);
            res = res.size() > s1.size()? res:s1;
            res = res.size() > s2.size()? res:s2;
        }
        return res;
    }
};