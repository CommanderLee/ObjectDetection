typedef struct{
	float weigth;
	unsigned short dimAndThr;
	unsigned short signAndChildindex;
} WeakClassifier;
const int g_boostSize = 200;
WeakClassifier g_data[200] = {
	{0.1,2,2},
	{0.1,2,2},
};