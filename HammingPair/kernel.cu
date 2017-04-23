
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

#define SIZE 2000
#define LENGTH 1000

using namespace std;
using namespace thrust;




class device_string
{
private:

public:
	int cstr_len;
	int ones=-1;
	char* raw;
	thrust::device_ptr<char> cstr;
	static char* pool_raw;
	static thrust::device_ptr<char> pool_cstr;
	static thrust::device_ptr<char> pool_top;


	__host__ static void init(int size=1024, int len=1024)
	{	 
		static bool v = true;
		if (v)
		{
			v = false;

			const int POOL_SZ = size*len;
			cout << "malloc\n";
			pool_cstr = thrust::device_malloc<char>(POOL_SZ*2);
			cout << "malloc end\n";
			pool_raw = (char*)raw_pointer_cast(pool_cstr);

			pool_top = pool_cstr;
		}
	}
	__host__ static void fini()
	{
		init();
		thrust::device_free(pool_cstr);
	}

	__host__ __device__ device_string(const device_string& s)
	{
		cstr_len = s.cstr_len;
		raw = s.raw;
		cstr = s.cstr;
	}

	__host__ __device__ void calcOnes(){
		ones = 0;
		for (size_t i = 0; i < cstr_len; i++)
		{
			if (cstr[i] == '1')
				ones++;
		}
	}

	__host__ device_string(const std::string& s)
		:
		cstr_len(s.length())
	{
		init();

		cstr = pool_top;
		pool_top += cstr_len + 1;
		raw = raw_pointer_cast(cstr);

		cudaMemcpy(raw, s.c_str(), cstr_len + 1, cudaMemcpyHostToDevice);
	}
	__host__ __device__ device_string()
		:
		cstr_len(-1),
		raw(0)
	{}



	__host__ operator std::string()
	{
		std::string ret;
		thrust::copy(cstr, cstr + cstr_len, back_inserter(ret));
		return ret;
	}
};
device_ptr<char> device_string::pool_cstr;
device_ptr<char> device_string::pool_top;
char* device_string::pool_raw;
__global__ void countKernel(int *pos, const int *ones, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
		atomicAdd(&pos[ones[i]], 1);

}

__device__ bool match(const device_string *words, int a, int b) {
	int c = 0;
	for (int i = 0; i < words[0].cstr_len; i++)
	{
		if (words[a].cstr[i] != words[b].cstr[i])
			c++;
		if (c > 1) return false;
	}
	return c == 1;
}

__global__ void findNeighborsKernel(device_string *words, int* ones, int *positions, int size, int len, int* res, int* cnt)
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



void readData(host_vector<device_string> &vec, int size, int wordLen) {
	
	vec = host_vector<device_string>(size);
	for (int i = 0; i < size-2; i++)
	{
		string s;
		for (int j= 0; j < wordLen; j++)
		{
			s.append(rand() % 2 ? "0" : "1");
		}
		vec[i] = device_string(s);
	}
	string s;
	for (int j = 0; j < wordLen; j++)
	{
		s.append("0");
	}
	vec[size-2] = device_string(s);
	s.replace(0, 1, "1");
	vec[size - 1] = device_string(s);
}
struct CalcOnes
{

	__device__ __host__ int operator()(device_string& s) {
		
		int ones = 0;
		for (size_t i = 0; i < s.cstr_len; i++)
		{
			if (s.cstr[i] == '1')
				ones++;
		}
		return ones;
		
	}
};

void calcVectors(device_vector<device_string> &d_words, device_vector<int> &ones, device_vector<int> &positions, int size) {

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

struct GetStr {
	__host__ __device__ char* operator()(const device_string &s) {
		return s.raw;
	}
};

void findPairs(device_vector<device_string> &d_words, device_vector<int> &ones, device_vector<int> &positions, int size) {
	int* d_pos = thrust::raw_pointer_cast(positions.data());
	device_string* str = raw_pointer_cast(d_words.data());
	int* d_ones = raw_pointer_cast(ones.data());
	cout << "AAAAAAAA\n";
	//findNeighborsKernel << <(size + 256) / 256, 256 >> > (str, d_ones, d_pos, size);
}
__device__ int counter;
__global__ void hammingKernel(char* str1, char* str2, int len)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < len && str1[i] != str2[i])
		atomicAdd(&counter, 1);
}

vector<std::pair<string,string>> cudaFindPairs(host_vector<device_string> &words, int size, int len) {
	device_vector<device_string> d_words = words;
	device_vector<int> ones = device_vector<int>(size);
	device_vector<int> positions = device_vector<int>(len);

	thrust::transform(d_words.begin(), d_words.end(), ones.begin(), CalcOnes());

	sort_by_key(ones.begin(), ones.end(), d_words.begin());
	int* d_pos = thrust::raw_pointer_cast(positions.data());
	device_string* str = raw_pointer_cast(d_words.data());
	int* d_ones = raw_pointer_cast(ones.data());
	device_vector<int> res = device_vector<int>(size * 2, 0);
	int* d_res = raw_pointer_cast(res.data());
	int h_cnt = 0;
	device_vector<int> cnt = device_vector<int>(1, 0);
	int* d_cnt = raw_pointer_cast(cnt.data());
	int N = 1024;
	countKernel << <(size + 256) / 256, 256 >> > (d_pos, d_ones, size);

	thrust::inclusive_scan(positions.begin(), positions.end(), positions.begin());
	device_vector<char*> d_str = device_vector<char*>(size);
	auto s = device_string::pool_raw;
	words = d_words;
	//d_words.clear();
	thrust::transform(d_words.begin(), d_words.end(), d_str.begin(), GetStr());
	//auto t = (&d_words[0]).get()->cstr; 
	//host_vector<char*> hh = d_str;
	host_vector<int> pairs;
	if (size > len) {
		findNeighborsKernel << <(size + 256) / 256, 256 >> > (str, d_ones, d_pos, size, len, d_res, d_cnt);
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
				hammingKernel << <(len + 256) / 256, 256 >> > (d_str[i], d_str[j], len);
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

vector<std::pair<string, string>> hostFindPairs(host_vector<device_string> &words, int size, int len) {
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
	host_vector<device_string> words;


	device_string::init(size,len);
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
	/*for each (auto p in v)
	{
		cout << p.first << ' ' << p.second << endl;
	}*/

    return 0;
}

