#include "../include/preprocess.h"

int minSubArrLen(int target, vector<int>& nums)
{
    int left = 0;
    int sum = 0;
    int minLen = INT_MAX;
    for(int right = 0; right < nums.size(); right++)
    {
        if(nums[right] == target){return 1;}
        sum += nums[right];
        while(sum >= target)
        {
            sum -= nums[left];
            minLen = min(minLen, right-left+1);
            left ++;
        }
    }
    return minLen == INT16_MAX ? 0:minLen;
}