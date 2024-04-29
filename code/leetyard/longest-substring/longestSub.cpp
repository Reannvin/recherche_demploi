#include "../include/preprocess.h"

class Solution{
public:
    int lengthOfLongestSubstring(string s){
        if(s.size() == 0){return 0;}
        int maxLen = 0;
        int left = 0;
        unordered_set<int> lookup;
        for(int right = 0; right < s.size();right++){
            while(lookup.find(s[right]) != lookup.end()){
                lookup.erase(s[left]);
                left ++;
            }
            lookup.insert(s[right]);
            maxLen = max(maxLen,right - left + 1);
        }
        return maxLen;
    }
};