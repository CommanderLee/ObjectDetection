#pragma once

#include <string>
using namespace std;

static class PathX
{
public:
	static wstring GetFileName(wstring path);
	static wstring GetFileNameNoExtension(wstring path);
	static wstring GetDirectory(wstring path);
	static wstring GetExtension(wstring path);
};

