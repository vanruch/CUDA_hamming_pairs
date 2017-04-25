
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "device_functions.h"


#include <cuda_runtime_api.h>

#include <device_launch_parameters.h>

#include <stdio.h>
#include <string>
#include <numeric>
#include <ctime>

#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>
#include <thrust\copy.h>
#include <thrust\device_ptr.h>
#include <thrust\device_malloc.h>
#include <thrust\for_each.h>
#include <thrust\transform.h>
#include <thrust/execution_policy.h>
#include <thrust\scan.h>

#define SIZE 256
#define LENGTH 1000
#define SIZEOF (sizeof(uint32_t)*8)
using namespace std;
using namespace thrust;




//class device_string
//{
//private:
//
//public:
//	int cstr_len;
//	int ones=-1;
//	char* raw;
//	thrust::device_ptr<char> cstr;
//	static char* pool_raw;
//	static thrust::device_ptr<char> pool_cstr;
//	static thrust::device_ptr<char> pool_top;
//
//
//	__host__ static void init(int size=1024, int len=1024)
//	{	 
//		static bool v = true;
//		if (v)
//		{
//			v = false;
//
//			const int POOL_SZ = size*len;
//			cout << "malloc\n";
//			pool_cstr = thrust::device_malloc<char>(POOL_SZ*2);
//			cout << "malloc end\n";
//			pool_raw = (char*)raw_pointer_cast(pool_cstr);
//
//			pool_top = pool_cstr;
//		}
//	}
//	__host__ static void fini()
//	{
//		init();
//		thrust::device_free(pool_cstr);
//	}
//
//	__host__ __device__ device_string(const device_string& s)
//	{
//		cstr_len = s.cstr_len;
//		raw = s.raw;
//		cstr = s.cstr;
//	}
//
//	__host__ __device__ void calcOnes(){
//		ones = 0;
//		for (size_t i = 0; i < cstr_len; i++)
//		{
//			if (cstr[i] == '1')
//				ones++;
//		}
//	}
//
//	__host__ device_string(const std::string& s)
//		:
//		cstr_len(s.length())
//	{
//		init();
//
//		cstr = pool_top;
//		pool_top += cstr_len + 1;
//		raw = raw_pointer_cast(cstr);
//
//		cudaMemcpy(raw, s.c_str(), cstr_len + 1, cudaMemcpyHostToDevice);
//	}
//	__host__ __device__ device_string()
//		:
//		cstr_len(-1),
//		raw(0)
//	{}
//
//
//
//	__host__ operator std::string()
//	{
//		std::string ret;
//		thrust::copy(cstr, cstr + cstr_len, back_inserter(ret));
//		return ret;
//	}
//};
class device_bitset {
private:
public:
	uint32_t* numbers;
	int bitLength,size;
	__host__  __device__ device_bitset()
	{

	}
	__host__ device_bitset(int length)
	{	
		cudaMalloc(&numbers, sizeof(uint32_t)*((length+ SIZEOF) / SIZEOF));
		uint32_t* h_numbers;
		h_numbers = (uint32_t*)malloc(sizeof(uint32_t)*((length + SIZEOF) / SIZEOF));
		bitLength = length;
		int i = 0;
		while (length > 0) {
			h_numbers[i++] = 0;
			length -= SIZEOF;
		}
		size = i;
		cudaMemcpy(numbers, h_numbers, sizeof(uint32_t)*((length + SIZEOF) / SIZEOF), cudaMemcpyHostToDevice);
	}

	

	__host__ device_bitset(string s)
	{
		bitLength = s.length();
		cudaMalloc(&numbers, sizeof(uint32_t)*((bitLength + SIZEOF) / SIZEOF));
		uint32_t* h_numbers;
		h_numbers = (uint32_t*)malloc(sizeof(uint32_t)*((bitLength + SIZEOF) / SIZEOF));
		int l = 0;
		for (int i = 0; i < s.length(); i+= SIZEOF)
		{
			uint32_t t = 0;
			for (int j = 0; j < SIZEOF; j++)
			{
				if (i + j >= s.length() || s[i + j] == '0') {
					t *= 2;
				}
				else {
					t = t * 2 + 1;
				}
			}
			
			h_numbers[l++] = t;
		}
		size = l;
		cudaMemcpy(numbers, h_numbers, sizeof(uint32_t)*((bitLength + SIZEOF) / SIZEOF), cudaMemcpyHostToDevice);
	}
	__device__ __host__ int bitCount() {
		int r = 0;
		unsigned int uCount;
		for (int i = 0; i < size; i++)
		{
			uCount= numbers[i] - ((numbers[i] >> 1) & 033333333333) - ((numbers[i] >> 2) & 011111111111);
			r+= ((uCount + (uCount >> 3)) & 030707070707) % 63;
		}
		return r;
	}
	__host__ operator string() {
		device_ptr<uint32_t> p(numbers);
		device_vector<uint32_t> d_bits(p, p + size);
		host_vector<uint32_t> h_bits = d_bits;
		string s;
		s.resize(bitLength);
		for (int i = 0; i < bitLength; i++)
		{
			s[i] = '0' + (h_bits[i / SIZEOF] & (1 << (i % SIZEOF)));
		}
		return s;
	}

};
//__device__ device_bitset operator ^(const device_bitset &a, const device_bitset &b) {
//	device_bitset r(a.bitLength);
//	for (int i = 0; i < a.size; i++)
//	{
//		r.numbers[i] = a.numbers[i] ^ b.numbers[i];
//	}
//	return r;
//}
__device__ int xorBits(const device_bitset&a, const device_bitset &b){
	int c = 0, uCount;
	for (int i = 0; i < a.size; i++)
	{
		uint32_t u = a.numbers[i] ^ b.numbers[i];
		uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
		c += ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}
	return c;
}

__global__ void countKernel(int *pos, const int *ones, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
		atomicAdd(&pos[ones[i]], 1);

}

__device__ bool match(const device_bitset *words, int a, int b) {
	return xorBits(words[a],words[b]) == 1;
}

__global__ void findNeighborsKernel(device_bitset *words, int* ones, int *positions, int size, int len, int* res, int* cnt)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= size) return;
	int from = ones[i] == 0 ? 0 : positions[ones[i]];
	int to = ones[i] == len - 1 ? size : positions[ones[i] + 1];
	for (int j = from; j < to; j++)
	{
		if (match(words, i, j)) {
			res[*cnt] = i;
			res[*cnt + 1] = j;
			atomicAdd(cnt, 2);
		}
			
	}


}



void readData(host_vector<device_bitset> &vec, int size, int wordLen) {
	
	vec = host_vector<device_bitset>(size);
	for (int i = 0; i < size-2; i++)
	{
		string s;
		for (int j= 0; j < wordLen; j++)
		{
			s.append(rand() % 2 ? "0" : "1");
		}
		vec[i] = device_bitset(s);
	}
	string s;
	for (int j = 0; j < wordLen; j++)
	{
		s.append("0");
	}
	vec[size-2] = device_bitset(s);
	s.replace(0, 1, "1");
	vec[size - 1] = device_bitset(s);
}
struct CalcOnes
{

	__device__ __host__ int operator()(device_bitset& s) {
		
		return s.bitCount();
		
	}
};

void calcVectors(device_vector<device_bitset> &d_words, device_vector<int> &ones, device_vector<int> &positions, int size) {

	thrust::transform(d_words.begin(), d_words.end(), ones.begin(), CalcOnes());
	cout << "sorting" << endl;
	sort_by_key(ones.begin(), ones.end(), d_words.begin());

	int* d_pos = thrust::raw_pointer_cast(positions.data());

	int* d_ones = raw_pointer_cast(ones.data());
	int N = 1024;
	countKernel << <(size + 256) / 256, 256 >> > (d_pos, d_ones, size);
	
	thrust::inclusive_scan(positions.begin(), positions.end(), positions.begin());
}

struct Comp {
	__host__ __device__ bool operator()(const thrust::tuple<char, char> &t) {
		return t.get<0>() != t.get<1>();
	}
};

struct GetRawPtr {
	__host__ __device__ uint32_t* operator()(device_bitset &s){
		return s.numbers;
	}
};

//void findPairs(device_vector<device_bitset> &d_words, device_vector<int> &ones, device_vector<int> &positions, int size) {
//	int* d_pos = thrust::raw_pointer_cast(positions.data());
//	device_string* str = raw_pointer_cast(d_words.data());
//	int* d_ones = raw_pointer_cast(ones.data());
//	cout << "AAAAAAAA\n";
//	//findNeighborsKernel << <(size + 256) / 256, 256 >> > (str, d_ones, d_pos, size);
//}


__device__ int counter;
__global__ void hammingKernel(uint32_t* str1, uint32_t* str2, int len)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < len/sizeof(uint32_t) && str1[i] != str2[i]) {
		uint32_t u = str1[i] ^ str2[i];
		int uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
		atomicAdd(&counter, ((uCount + (uCount >> 3)) & 030707070707) % 63);
	}
}

vector<std::pair<string,string>> cudaFindPairs(host_vector<device_bitset> &words, int size, int len) {
	device_vector<device_bitset> d_words = words;
	device_vector<int> ones = device_vector<int>(size);
	device_vector<int> positions = device_vector<int>(len);

	thrust::transform(d_words.begin(), d_words.end(), ones.begin(), CalcOnes());

	sort_by_key(ones.begin(), ones.end(), d_words.begin());
	int* d_pos = thrust::raw_pointer_cast(positions.data());
	device_bitset* bits = raw_pointer_cast(d_words.data());
	int* d_ones = raw_pointer_cast(ones.data());
	device_vector<int> res = device_vector<int>(size * 2, 0);
	int* d_res = raw_pointer_cast(res.data());
	int h_cnt = 0;
	device_vector<int> cnt = device_vector<int>(1, 0);
	int* d_cnt = raw_pointer_cast(cnt.data());
	int N = 1024;
	countKernel << <(size + 256) / 256, 256 >> > (d_pos, d_ones, size);

	thrust::inclusive_scan(positions.begin(), positions.end(), positions.begin());
	device_vector<uint32_t*> d_bts = device_vector<uint32_t*>(size);
	//auto s = device_string::pool_raw;
	words = d_words;
	//d_words.clear();
	thrust::transform(d_words.begin(), d_words.end(), d_bts.begin(), GetRawPtr());
	//auto t = (&d_words[0]).get()->cstr; 
	//host_vector<char*> hh = d_str;
	host_vector<int> pairs;
	if (size > len) {
		findNeighborsKernel << <(size + 256) / 256, 256 >> > (bits, d_ones, d_pos, size, len, d_res, d_cnt);
		host_vector<int> hcnt = cnt;
		h_cnt = hcnt[0];
		pairs = res;
	}
	else {
		host_vector<int> h_ones = ones;
		host_vector<int> h_pos = positions;

		pairs = res;

		//words = d_words;
		int h_counter = 0;
		for (size_t i = 0; i < size; i++)
		{

			int from = h_ones[i] == 0 ? 0 : h_pos[h_ones[i]];
			int to = h_ones[i] == len - 1 ? size : h_pos[h_ones[i] + 1];


			for (size_t j = from; j < to; j++)
			{
				/*c = thrust::count_if(
				thrust::make_zip_iterator(thrust::make_tuple(s+i*len, s+j*len)),
				thrust::make_zip_iterator(thrust::make_tuple(s+(i*len + len), s+(j*len + len))),
				Comp()
				);*/
				h_counter = 0;
				cudaMemcpyToSymbol(counter, &h_counter, sizeof(int));
				hammingKernel << <(len/sizeof(uint32_t) + 256) / 256, 256 >> > (d_bts[i], d_bts[j], len);
				//cudaThreadSynchronize();
				cudaMemcpyFromSymbol(&h_counter, counter, sizeof(int));
				if (h_counter == 1) {
					pairs[h_cnt] = i;
					pairs[h_cnt] = j;
					h_cnt += 2;
				}
			}
		}
	}

	std::vector<std::pair<string, string>> p;
	//words = d_words;
	for (size_t i = 0; i < h_cnt; i += 2)
	{
		p.push_back(std::make_pair((string)words[pairs[i]], (string)words[pairs[i + 1]]));
	}

	return p;
}

vector<std::pair<string, string>> hostFindPairs(host_vector<device_bitset> &words, int size, int len) {
	vector<std::pair<string, string>> res;
	vector<string> w;
	for each (auto s in words)
	{
		w.push_back((string)s);
	}
	for (int i = 0; i < size; i++)
	{
		for (int j = i + 1; j < size; j++)
		{

			int c = 0;
			for (int k = 0; k < len; k++)
			{
				if (w[i][k] != w[j][k])
					c++;
				if (c > 1)
					break;
			}
			if (c == 1) {
				//res.push_back(std::make_pair(w[i], w[j]));
			}

		}
	}
	return res;
}


int main()
{

	int size = SIZE;
	int len = LENGTH;
	host_vector<device_bitset> words;


	
	cout << "gen\n";
	readData(words, size, len);
	cout << "calc 1\n";
	clock_t begin = clock();
	auto v = cudaFindPairs(words, size, len);
	clock_t end = clock();
	cout << "CUDA: " << double(end - begin) / CLOCKS_PER_SEC<<endl;
	begin = clock();
	v = hostFindPairs(words, size, len);
	end = clock();
	cout << "HOST: " << double(end - begin) / CLOCKS_PER_SEC << endl;
	for (int i = 0; i < words.size(); i++) {
		cudaFree(words[i].numbers);
	}
	/*for each (auto p in v)
	{
		cout << p.first << ' ' << p.second << endl;
	}*/

    return 0;
}

