#include<bits/stdc++.h>
using namespace std;
int main(){
	string string str = "151-152x(50/40+15)-10";
	string pat = "[()x+-/]";
	rgx(pat);
	string result;
	regex_replace(back_inserter(result),str.begin(),str.end(),rgx," $0 ");
	cout<<result<<endl;
	return 0;
}
