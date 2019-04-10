#include<bits/stdc++.h>
using namespace std;
bool isHuiwen(string ss,int s,int e){//[s,e] is huiwen or not
	int i=s,j=e;
	while(j-i!=0 && j-i!=-1){
		if(ss[i]!=ss[j])break;
		else{
			i++;
			j--;
		}
	}
	if(j-i==0 || j-i==-1){return true;}
	return false;
}
int findLastEqual(string s,int a,int b){//between [a,b),find the last same char as s[a] after a of index,if not,return -1
	int i=b-1;
	while(i!=a){
		if(s[i]==s[a])break;
		i--;
	}
	if(i!=a)return i;
	return -1;
}
int main(){
	string str;
	cin>>str;
	int a=0,b;
	int maxLen=0;//record the max length of huiwen found ever
	int result = -1;//record the begin of huiwen
	while(a!=str.size()-1 && str.size()-a>maxLen){//if the length of rest substr is shorter than maxLen, then we don't need to find anymore
		b=str.size();

		//as for the fixed a(beginning),we see if we can find the huiwen
		b=findLastEqual(str,a,b);
		if(b!=-1){//if b==-1,then fail,we just move a,then next loop
			while(b!=-1 && !isHuiwen(str,a,b)){
				b=findLastEqual(str,a,b);
			}
			if(b!=-1){//if we find huiwen
				if(b-a+1>maxLen){
					maxLen=b-a+1;
					result=a;
				}
			}
		}

		a++;
	}
	cout<<result<<" "<<maxLen<<endl;
}
