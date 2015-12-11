#ifndef BOOST_PREDICT_H_
#define BOOST_PREDICT_H_

#ifdef __cplusplus
extern "C"{
#endif

typedef struct{
	float weigth;
	unsigned short dimAndThr;
	unsigned short signAndChildindex;
} WeakClassifier;

#ifdef __cplusplus
};
#endif	//extern "C"
#endif	//BOOST_PREDICT_H_