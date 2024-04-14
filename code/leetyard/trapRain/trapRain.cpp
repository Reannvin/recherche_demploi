#include "../include/preprocess.h"

int trapDoublePointer(vector<int>& height)
{
    if(height.size() <= 2){return 0;}
    vector<int> maxLeft(height.size(), 0);
    vector<int> maxRight(height.size(), 0);
    int size = maxRight.size();

    maxLeft[0] = height[0];
    for(int i = 1; i < size; i++)
    {
        maxLeft[i] = max(height[i], maxLeft[i-1]);
    }

    maxRight[size - 1] = height[size - 1];
    for(int i = 1; i < size; i++)
    {
        maxLeft[i] = max(height[i], maxLeft[i-1]);
    }

    int sum = 0;
    for(int i = 0;i<size;i++)
    {
        int count = min(maxLeft[i], maxRight[i]) - height[i];
        if (count > 0) sum += count;
    }
    return sum;
}

int trapStack(vector<int>& height)
{
    if(height.size() <= 2) return 0;
    stack<int> st;
    st.push(0);
    int sum = 0;
    for(int i = 1;i < height.size(); i++)
    {
        while(!st.empty() && height[i] > height[st.top()])
        {
            int mid = st.top();
            st.pop();
            if(!st.empty())
            {
                int h = min(height[st.top()], height[i]) - height[mid];
                int w = i - st.top() - 1;
                sum = h * w;
            }
        }
        st.push(i);
    }
    return sum;
}