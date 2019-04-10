#include<algorithm>
#include<iostream>
using namespace std;

int graph[105][105];
int C,S,Q;
const int INF=0x3fffffff;
int main(){
	int time = 1;
	while(true){
		cin>>C>>S>>Q;
		if(C==0 && S==0 && Q==0)
			break;
		cout<<"Case #"<<time<<endl;
		for(int i=1;i<C+1;++i)
			for(int j=0;j<C+1;++j){
				if(i==j)graph[i][j]=0;
				else graph[i][j]=INF;
			}
		int a,b,c;
		while(S--){
			cin>>a>>b>>c;
			graph[a][b]=graph[b][a]=c;
		}
		for(int k=1;k<C+1;++k)
			for(int i=1;i<C+1;++i)
				for(int j=1;j<C+1;++j)
					graph[i][j]=min(graph[i][j],max(graph[i][k],graph[k][j]));
		while(Q--){
			cin>>a>>b;
			if(graph[a][b]==INF)
				cout<<"no path"<<endl;
			else
				cout<<graph[a][b]<<endl;
		}
		++time;
	}
	return 0;
}
