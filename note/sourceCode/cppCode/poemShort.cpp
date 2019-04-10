#include<bits/stdc++.h>
using namespace std;
typedef struct node{
	int data;
	int weight;
	int codeLen;//length of code
	struct node* left, *right;
	node(int dat,int wt,node* lft=nullptr,node* rt=nullptr){
		data = dat;
		weight = wt;
		left = lft;
		right = rt;
	}
}node;

node* create(node** nodes,int s,int e){
	int weight = 0;
	for(int i=s;i<=e;++i)weight+=nodes[i]->weight;
	node* root = new node(-1,weight);
	root->left = nodes[s];
	node* nd = root->left;
	for(int i=s+1;i<=e;++i){
		nd->right = nodes[i];
		nd = nd->right;
	}
	return root;
}

void sort(node** nodes,int s,int e){
	if(e-s>10){//if the lenghth is too long,we should use quick sort
		int i=s,j=e;
		while(i<j){
			while(i<=e && nodes[i]->weight<=nodes[s]->weight){
				++i;
			}
			while(j>=s && nodes[j]->weight>=nodes[s]->weight){
				--j;
			}
			if(i<j){
				node* temp = nodes[i];
				nodes[i] = nodes[j];
				nodes[j] = temp;
			}
		}
		if(j>s){//j is possible to be s-1
			node* temp = nodes[j];
			nodes[j] = nodes[s];
			nodes[s] = temp;
		}
		sort(nodes,s,j-1);
		sort(nodes,i,e);
	}
	if(e-s>0 && e-s<=10){//bubble sort
		int bound = e;
		while(bound != s){
			int t = s;
			for(int i=s;i<bound;++i){
				if(nodes[i]->weight>nodes[i+1]->weight){
					node* temp = nodes[i];
					nodes[i] = nodes[i+1];
					nodes[i+1] = temp;
					t = i;
				}
			}
			bound = t;
		}
	}
}

void deleteRoot(node* nd){
	if(nd->left==nullptr && nd->right==nullptr){
		delete nd;
	}
	else{
		if(nd->left)deleteRoot(nd->left);
		if(nd->right)deleteRoot(nd->right);
	}
}
int main(){
	int n,k;
	cin>>n>>k;
	node** nodes = new node*[n];
	for(int i=0;i<n;++i){
		int weight;
		cin>>weight;
		node* nd = new node(i,weight);
		nodes[i] = nd;
	}

	int a=0;
	while(a != n-1){
		//first,sort the nodes array
		sort(nodes,a,n-1);

		if(a+k-1<n){//if the rest is enough to k
			nodes[a+k-1]=create(nodes,a,a+k-1);
			a = a+k-1;
		}
		else{
			nodes[n-1] = create(nodes,a,n-1);
			a = n-1;
		}
	}

	//broad search the haffman tree
	queue<node*> que;
	node* root = nodes[n-1];
	root->codeLen = 0;
	que.push(root);
	int maxLen = 0;//record the max length of code
	int totalLen = 0;
	while(!que.empty()){
		root = que.front();
		que.pop();
		int len = root->codeLen;
		if(!root->left){//if it is leaf node,then add it to totalLen of code
			totalLen += len*root->weight;
		}
		root = root->left;
		while(root){
			root->codeLen = len+1;
			que.push(root);
			root = root->right;
			if(len+1>maxLen)maxLen=len+1;
		}
	}

	cout<<totalLen<<" "<<maxLen<<endl;
	deleteRoot(nodes[n-1]);
	delete[] nodes;
	
	
	return 0;
}
