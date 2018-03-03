void serial(int* a, int n){
  int is_sorted = 1;
  do{
    is_sorted = 1;
    for(int i=0;i<n;i+=2)
      if (a[i]>a[i+1]){
	swap(a, i, i+1);
	is_sorted = 0;
      }
    for(int i=1;i<n;i+=2)
      if (a[i]>a[i+1]){
	swap(a, i, i+1);
	is_sorted = 0;
      }
  } while(is_sorted==0);
}
