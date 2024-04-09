#include <vector>
#include <iostream>

struct ListNode
{
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

ListNode* reverseList(ListNode* head)
{
    if(head ==  nullptr || head->next == nullptr){return head;}
    ListNode* curr = reverseList(head->next);
    head->next->next = head;
    head->next = nullptr;
    return curr;
}


