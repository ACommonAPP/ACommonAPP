//I will practice bubble sort below
/*#include<iostream>
using namespace std;

int main(){
	int arr1[5];
	for(int i=0;i<5;++i){
		cin>>arr1[i];
	}
	
	int bound = 5-1;
	while(bound != 0){
		int t = 0;
		for(int j=0;j<bound;++j){
			if(arr1[j] > arr1[j+1]){
				int temp = arr1[j];
				arr1[j] = arr1[j+1];
				arr1[j+1] = temp;
				t = j;
			}
		}
		bound = t;
	}

	//print the result of bubble sorting
	for(int i=0;i<5;++i){
		printf("%d",arr1[i]);
		printf("\n");
	}
}*/



//I will practice quick sort below
/*#include<iostream>
using namespace std;

//accompished by recurse
void quickSort(int arr[],int i,int j){
	int m = i, n = j+1;
	if(i<j){
		while(m<n){
			m++;			
			while(m<=j && arr[m]<=arr[i]){
				m++;
			}
			n--;			
			while(n>=i && arr[n]>=arr[i]){
				n--;
			}

			if(m<n){
				int temp = arr[m];
				arr[m] = arr[n];
				arr[n] = temp;
			}
		}
		if(n>i){
			int temp = arr[i];
			arr[i] = arr[n];
			arr[n] = temp;
		}

		quickSort(arr,m,j);
		quickSort(arr,i,n-1);
	}	
}
int main(){
	int arr[10] = {27,99,0,8,13,64,86,16,7,10};
	quickSort(arr,0,9);
	for(int i=0;i<10;++i){
		cout<<arr[i]<<" ";
	}
	cout<<endl;
}*/


//practice  heapSort
#include<bits/stdc++.h>
using namespace std;
void restore(int* R,int f,int e){//重建根为Rf的二叉树，使之满足堆的特性，且这个树的任意节点，编号都不大于e
	int j=f;
	while(j<=(e-1)/2 && f!=e){//当j=e=0时,(e-1)/2=0,则本来排序好的就又反过来了
		int m;//m是R[j]的较大的子结点
		if(2*j+2<=e && R[2*j+1]<R[2*j+2]){
			m=2*j+2;
		}
		else{
			m=2*j+1;
		}
		if(R[m]>R[j]){
			int temp=R[m];
			R[m]=R[j];
			R[j]=temp;
			j=m;
		}
		else{
			j=e;
		}
	}
}

int main(){
	int R[20];
	for(int i=0;i<20;++i){
		R[i]=20-i;
	}
	for(int i=(20-2)/2;i>=0;--i){
		restore(R,i,20-1);
	}
	for(int i=20-1;i>=1;--i){
		cout<<"max="<<R[0]<<endl;
		int temp=R[i];
		R[i]=R[0];
		R[0]=temp;
		restore(R,0,i-1);
	}
	for(int i=0;i<20;++i){
		cout<<R[i]<<endl;
	}
}
