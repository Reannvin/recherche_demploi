#include "../include/preprocess.h"

bool WINDOW_NEEDS_SHRINK = 1;

void slidingWindow(string s, string t){
    unordered_map<char, int> window;
    int left = 0, right = 0;
    while(right < s.size()){
        char c = s[right];
        right ++;
        while(WINDOW_NEEDS_SHRINK){
            char d = s[left];
            left++;
        }
    }
}