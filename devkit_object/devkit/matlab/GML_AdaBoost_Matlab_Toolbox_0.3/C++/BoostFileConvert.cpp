#include "stdafx.h"
#include "BoostFileConvert.h"
#include "BoostedCommittee.h"
#include <fstream>

void BoostFileConvert(const wstring& srcFilePath, const wstring& saveFileName)
{
	CBoostedCommittee boost;
	//boost.LoadFromFile(srcFilePath.c_str());

	ofstream out(saveFileName);
	
	out.flush();
	out.close();

}




