
public class FirstApp{
	public static void main(String args[]){
		SingleTon instance = SingleTon.getInstance(1);
	}   
}

class SingleTon{
	private int type;
	private SingleTon(int i){
		type = i;
		System.out.println("create instance"+i);
	}
	private final static SingleTon INSTANCE = null;
    static SingleTon getInstance(int i){
		if(INSTANCE == null){
			INSTANCE =new SingleTon(i);
		}
		return INSTANCE;
	}
}
