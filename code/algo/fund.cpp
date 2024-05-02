#include <iostream>
#include <vector>

using namespace std;

int main()
{
    vector<int> array;
    
    int i = 3;
    // 在末尾增加
    array.push_back(i);

    // 在中间插入
    array.insert(array.begin(),1);

    // 删除末尾
    array.pop_back();

    // 删除元素
    array.erase(array.begin());


    return 0;
}