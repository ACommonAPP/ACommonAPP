#include<iostream>
#include<string>
#include<regex>
#include<vector>
#include<stack>
#include<map>
using namespace std;
typedef struct node{
	string value;
	struct node* left;
	struct node* right;
}node;
node* create(vector<string> vec,int s,int e){//create a small binary tree from expression given
	//first,create a suffixed expression
	int a=0,size=vec.size();//a record where we are in the vec
	vector<string> vec1;//store the suffix expression
	stack<string> stk;
	map<string,int> map1;
	map1["+"]=0;map1["-"]=0;map1["x"]=1;map1["/"]=1;
	while(a != size){
		if(vec[a].compare("+")==0 || vec[a].compare("-")==0 || vec[a].compare("x")==0 || vec[a].compare("/")==0){
			if(stk.empty())
				stk.push(vec[a]);
			else{
				while(!stk.empty() && stk.top().compare("(")!=0 && map1[stk.top()]>=map1[vec[a]]){
					vec1.push_back(stk.top());
					stk.pop();
				}
				stk.push(vec[a]);
			}
		}
		else if(vec[a].compare("(")==0){
			stk.push(vec[a]);
		}
		else if(vec[a].compare(")")==0){
			while(stk.top().compare("(")!=0){
				vec1.push_back(stk.top());
				stk.pop();
			}
			stk.pop();
		}
		else{
			vec1.push_back(vec[a]);
		}
		a++;
	}
	while(!stk.empty()){
		vec1.push_back(stk.top());
		stk.pop();
	}
	
	//transform the suffixed expression to binery expression tree
	stack<node*> stkNode;
	int b=0;
	size=vec1.size();
	while(b != size){
		if(vec1[b].compare("+")==0 || vec1[b].compare("-")==0 || vec1[b].compare("x")==0 || vec1[b].compare("/")==0){
			node* root = new node();
			root->value = vec1[b];
			root->right = stkNode.top();stkNode.pop();
			root->left = stkNode.top();stkNode.pop();
			stkNode.push(root);
		}
		else{
			node* root = new node();
			root->value = vec1[b];
			root->left = root->right = nullptr;
			stkNode.push(root);
		}
		b++;
	}
	return stkNode.top();
}
int transform(string s){//transform a string to int, not for "+" string, it's for "122"
	int a = 0;//get the length of string
	int sum = 0;
	while(a != s.length()){
		sum = sum*10 + s[a] - 48;
		a++;
	}
	return sum;
}
void deleteRoot(node* root){//delete the root
	if(root->left==nullptr && root->right==nullptr){
		delete root;
	}
	else{
		deleteRoot(root->left);
		deleteRoot(root->right);
		delete root;
	}
}
float compute(node* root){//compute the value of expression according to the root
	if(root->left==nullptr && root->right==nullptr){
		return transform(root->value);
	}
	else{
		if(root->value.compare("+")==0){
			return compute(root->left)+compute(root->right);
		}
		else if(root->value.compare("-")==0){
			return compute(root->left)-compute(root->right);
		}
		else if(root->value.compare("x")==0){
			return compute(root->left)*compute(root->right)*1.0;
		}
		else{
			return compute(root->left)/(compute(root->right)*1.0);
		}
	}
}
int main(){
	string s;
	cout<<"please input the expression string:"<<endl;
	getline(cin,s);
	//split the expression string
	string str = "[()x+-/]";//I don't know if it is right, I have to test it later???
	regex pat(str);
	string sResult;
	regex_replace(back_inserter(sResult),s.begin(),s.end(),pat," $0 ");//I don't know???
	cout<<"result ="<<endl;
	cout<<sResult<<endl;
	//split the string s to vec
	vector<string> vec;
	int i=0,j=0, n=0,size=sResult.size();
	while(sResult[i]==' ')++i;
	while(i != size){
		j = i;
		while(j!=size && sResult[j]!=' '){
			j++;
		}
		vec.push_back(sResult.substr(i,j-i));
		while(j!=size && sResult[j]==' ')++j;
		i = j;
	}

	node* root = create(vec,0,vec.size()-1);
	cout<<"the result is "<<compute(root)<<endl;
	deleteRoot(root);
	return 0;
}
