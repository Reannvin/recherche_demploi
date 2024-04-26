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

TreeNode* find(TreeNode* root, int val1, int val2){
    if(root == nullptr){return nullptr;}
    if(root->val == val1 || root->val ==val2){
        return root;
    }
    TreeNode* left = find(root->left,val1,val2);
    TreeNode* right = find(root->right,val1,val2);
    if(left != nullptr && right != nullptr){
        return root;
    }
    return left != nullptr ? left : right;
}

TreeNode* lowestCommonAncestor(TreeNode* root,TreeNode* p, TreeNode* q){
    return find(root,p->val,q->val);
}
