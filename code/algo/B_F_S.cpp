#include <iostream>
#include <vector>
#include <queue>
#include <set>


using namespace std;

int BFS(Node start, Node target) {
    queue<Node> q;
    set<Node> visited;

    q.push(start);
    visited.insert(start);

    while(!q.empty()){
        int size = q.size();
        for(int i = 0; i < size; i++){
            Node cur = q.front();
            q.pop();
            if(cur == target){
                return step;
            }
            for (Node x : cur.adj()){
                if (visited.count(x) == 0){
                    q.push(x);
                    visited.insert(x);
                }
            }
        }
    }
}