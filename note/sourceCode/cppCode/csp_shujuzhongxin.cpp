#include<bits/stdc++.h>
using namespace std;
struct Edge{
	int v1,v2,cost;
	Edge(int vv1.int vv2.int c):v1(vv1),v2(vv2),cost(c){};
	//下面的语句是为了改变priority_queue是大端还是小端
	bool operator <(const Edge&e)const{
		return this->cost>e.cost;
	}
}
int findFather(int* father,int x){
	
}
int main(){
	priority_queue<Edge> edges;
	int n,m,root,ans=0;//ans记录最小生成树的最大边
	cin>>n>>m>>root;
	int* father = new int[n+1];//记录存在father[1]-father[n]
	while(m--){
	eee	int a,b,c;
		cin>>a>>b>>c;
		edges.push(Edge(a,b,c));
	}
	while(!edges.empty()){
		
	}
}

