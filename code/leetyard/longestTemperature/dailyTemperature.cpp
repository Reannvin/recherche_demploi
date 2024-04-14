#include "../include/preprocess.h"

vector<int> dailyTemp(vector<int>& T)
{
    stack<int> st;
    vector<int> result(T.size(),0);
    for (int i = 0; i < T.size(); i++)
    {
        while (!st.empty() && T[i] > T[st.top()])
        {
            result[st.top()] = i - st.top();
            st.pop();
        }
        st.push(i);
    }
    return result;
}