#ifndef GPU_THRUST_CUH
#define GPU_THRUST_CUH

void maxReduction(int* , int&, int&, int);
void memtest();
#endif // !GPU_THRUST_CUH

void minMaxRWrapper(int *&iU, int *&iV, int iNU, int iNV,
				    int* &iminU, int* &imaxU,
					int* &iminV, int* &imaxV);