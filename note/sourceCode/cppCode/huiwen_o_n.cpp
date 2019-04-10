
#include<bits/stdc++.h>
using namespace std;
int findLen(string s,int i,int n){//根据位置i找到以i为中心的huiwen的左半部分长度,n为str的长度
	int a=1;
	while(i-a>=0 && i+a<=n-1){
		if(s[i-a] != s[i+a]){
			--a;
			break;
		}
		++a;
	}
	if(i-a<0 || i+a>n-1)--a;
	if(s[i-a]=='#')--a;
	return a;
}
int main(){
	string str;
	cin>>str;
	//通过加符号的方式将str串变成奇数串
	int n=2*str.size()+1;
	str.resize(n+1);
	for(int i=(n-1)/2-1;i>=0;i--){
		str[2*i+1]=str[i];
		str[2*i+2]='#';
	}
	str[0]='#';
	str[2*n+1]='\0';
	
	int* len = new int[n];//记录huiwen的左半部分的长度
	int p = 0;//p记录目前最长huiwen的中心点
	for(int i=0;i<n;++i){
		if(i==0){
			len[i]=findLen(str,i,n);
		}
		else{
			int j=2*p-i;
			if(j<0 || i>=p+len[p]){//???
				len[i]=findLen(str,i,n);
			}
			else if(len[j]<p+len[p]-i){//???
				len[i]=len[j];
			}
			else{
				int a=len[j]+1;
				while(i-a>=0 && i+a<=n-1){
					if(str[i-a] != str[i+a]){
						--a;
						break;
					}
					++a;
				}
				if(i-a<0 || i+a>n-1)--a;
				if(str[i-a]=='#')--a;
				len[i]=a;
			}
			if(len[i]>len[p])p=i;
		}
	}

	cout<<(p-len[p]-1)/2<<" "<<len[p]+1<<endl;
}
