/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

Appagato is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
template<int BLOCKF, int MIN_VALUE, int MAX_VALUE>
__device__ __forceinline__ int logValue(int Value) {
	int logSize = 31 - __clz(BLOCKF / Value);
	if (logSize < LOG2<MIN_VALUE>::value)
		logSize = LOG2<MIN_VALUE>::value;
	if (logSize > LOG2<MAX_VALUE>::value)
		logSize = LOG2<MAX_VALUE>::value;
	return logSize;
}

/*
struct compareFloatABS_Struct {           // function object type:
	float Epsilon;
	compareFloatABS_Struct(float _Epsilon) : Epsilon(_Epsilon) {}

	inline bool operator() (const float a, const float b) {
		return fabs(a - b) < Epsilon;
	}
};

struct compareFloatRel_Struct {           // function object type:
	float Epsilon;
	compareFloatRel_Struct(float _Epsilon) : Epsilon(_Epsilon) {}

	inline bool operator() (const float a, const float b) {
		float diff = fabs(a - b);
		return (diff < Epsilon) || (diff / std::max(fabs(a), fabs(b)) < Epsilon);
	}
};*/
