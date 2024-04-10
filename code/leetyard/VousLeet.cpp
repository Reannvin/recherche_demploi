#include <iostream>
#include <vector>
#include <stack>

using namespace std;

// 给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。

class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res(n, vector<int>(n,0));
        int startx = 0, starty = 0;
        int loop = n / 2;
        int mid = n / 2;
        int count = 1;
        int offset = 1;
        int i,j;
        while(loop--)
        {
            i = startx;
            j = starty;
            for(j;j < n - offset;j++){res[i][j] = count++;}
            for(i;i < n - offset;i++){res[i][j] = count++;}
            for(;j > starty;j--){res[i][j] = count++;}
            for(;i > startx;i--){res[i][j] = count++;}
            startx++;
            starty++;
            offset += 1;
        }
        if(n%2){res[mid][mid] = count;}
        return res;
    }
};