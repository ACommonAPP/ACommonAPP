#include<iostream>
#include<stack>
#include<vector>
using namespace std;
int edge[101][101];//store the edge information,edge[i][0] means how many vertex is connected to i,if edge[i][j]=-1,it means i and j are not connected
int visited[101][101];//record if the crossing is visited,visited[v][t] where t means the depth of v;v visited can be in different depth, for example, the first time visited v or the second time visit v; 
void initEdge(){
	for(int i=0;i<101;++i){
		edge[i][0] = 0;
		for(int j=1;j<101;++j)
			edge[i][j] = -1;
	}
}
void initVisited(){
	for(int i=0;i<101;++i){
		for(int j=0;j<101;++j){
			visited[i][j] = 0;
		}
	}
}
bool func1(int a,int t){//check if visited[a][1] to visited[a][t] is 1
	while(t != 0){
		if(visited[a][t]==1)
			return true;
		t--;
	}
	return false;
}

int main(){
	int c,s,q;
	while(true){
		cin>>c>>s>>q;
		if(c==0 && s==0 && q==0){
			break;
		}
		initEdge();//first,initial the edge array
		//construct the edge map
		for(int i=0;i<s;++i){
			int a,b,intensity;
			cin>>a>>b>>intensity;
			edge[a][b] = intensity;
			++edge[a][0];
			edge[b][a] = intensity;
			++edge[b][0];
		}

		for(int i=0;i<q;++i){
			int a,b;//store the crossings where you ask for the min decibels
			cin>>a>>b;

	   		initVisited();
			int v=a,t=1,max=-1;//max means the max fenbei of the exact path from a to b
			vector<int> vec;
			stack<int> stk;
			stack<int> stk1;//record the path and max
			stk.push(v);stk.push(t);visited[v][t] = 1;
			stk1.push(v);stk1.push(max);
			while(!stk.empty()){
				int t1 = stk.top();stk.pop();//t1 is temporarily for the t value,which is used for stk1 thing
				int v1 = stk.top();stk.pop();
				if(v1 != v){//update the max value
					int count = t - t1 + 1;
					while(count--){//v is regarded the last crossing and v1 is regarded as the present one
						stk1.pop();
						int v2 = stk1.top();
						stk1.pop();
						for(int j=1;j<101;++j)//clear the visited record of history path which is  useful,even harmful to the present path
							visited[v2][j]=0;
						
					}
					max = stk1.top();stk1.pop();
					v = stk1.top();stk1.push(max);
					if(edge[v][v1]>max){
						max = edge[v][v1];
					}
					stk1.push(v1);stk1.push(max);
					cout<<"test line"<<endl;
					cout<<v1<<" "<<max<<endl;
				}
				t = t1;//update t
				v = v1;//update the v value


				
				if(v==b){//if we get to the end crossing,we must push the max value into vector
					vec.push_back(max);
				}
				else{//update the stack value
					for(int j=1;j<101;++j){
						if(edge[v][j]!=-1 && !func1(j,t)){
							stk.push(j);stk.push(t+1);
							visited[j][t+1] = 1;
						}
					}
				}
			}
			//get the min value of all vec values
			if(!vec.empty()){
				int min=vec.front();
				vector<int>::iterator it;
				for(it=vec.begin();it!=vec.end();it++){
					if(*it<min)
						min = *it;
				}
				cout<<min<<endl;
			}
			else
				cout<<"no path"<<endl;
		}
	}
}
