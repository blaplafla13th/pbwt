#include "util.h"
#include <vector>
#include <fstream>
#include <iterator>
#include <bits/stdc++.h>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace std;
using namespace boost::iostreams;
using namespace chrono;

void parallel_run(int argc, char *argv[]) {
    string f = argv[1];
    printf("Read file: %s\n", f.c_str());
    vector<vector<int> > X = read_hap(f);

    const int retry = 20;
    const int L = 4;
    printf("Test size: %lu haps x %lu sites\n", X.size(), X[0].size());
    printf("Retry: %d\n", retry);
    printf("Matched length: %d\n", L);

    int nthreads = static_cast<int>(thread::hardware_concurrency());
    printf("Max thread:%d\n\n", nthreads);
    for (int i = 2; i <= nthreads; ++i) {
        printf("Num thread:%d\n", i);
        signed long int duration_total_run = 0;
        high_resolution_clock::time_point start, stop;
        vector<vector<int> > matches;
        vector<vector<vector<int> > > res;
        for (int j = 0; j < retry; ++j) {
            start = high_resolution_clock::now();
            matches = build_and_match(X, i, L);
            stop = high_resolution_clock::now();
            duration_total_run += duration_cast<microseconds>(stop - start).count();
        }
        // print_matches(matches);
        printf("Time: %ld us\n\n", duration_total_run / retry);
    }
}

void single_run(int argc, char *argv[]) {
    string f = argv[1];
    printf("Read file: %s\n", f.c_str());
    vector<vector<int> > X = read_hap(f);

    const int retry = 20;
    const int L = 4;
    printf("Test size: %lu haps x %lu sites\n", X.size(), X[0].size());
    printf("Retry: %d\n", retry);
    printf("Matched length: %d\n", L);
    int nthreads = static_cast<int>(thread::hardware_concurrency());
    printf("Max thread:%d\n\n", nthreads);

    printf("Num thread:%d\n", 1);
    signed long int duration_total_run = 0;
    high_resolution_clock::time_point start, stop;
    vector<vector<int> > matches;
    vector<vector<vector<int> > > res;
    for (int j = 0; j < retry; ++j) {
        start = high_resolution_clock::now();
        matches = build_and_match(X, 1, 4);
        stop = high_resolution_clock::now();
        duration_total_run += duration_cast<microseconds>(stop - start).count();
    }
    // print_matches(matches);
    printf("Time: %ld us\n\n", duration_total_run / retry);
}

void print_matches(const vector<vector<int> > &all_matches_report)
{
   vector<vector<int> > sorted =(all_matches_report);

    // Sort the filtered matches by column 1 (position)
    sort(sorted.begin(), sorted.end(),
         [](const vector<int>& a, const vector<int>& b) {
             return a[0] == b[0] ? a[1] == b[1] ? a[2] < b[2] : a[1] < b[1] : a[0] < b[0];
         });

    // Print the sorted matches
    for (const auto &match: sorted) {
        printf("Position %d: [%d, %d], length=%d\n",
               match[0], match[1]+1, match[2]+1, match[3]);
    }
}

vector<vector<int> > read_hap(const string &filename) {
    if (filename.find(".gz") != string::npos)
        return read_hap_gz(filename);
    return read_hap_normal(filename);
}

vector<vector<int> > read_hap_normal(const string &filename) {
    vector<vector<int> > res;
    ifstream ifs(filename);       // open the file
    string tempstr;
    while (getline(ifs, tempstr)) {
        stringstream lineStream(tempstr);
        vector<int> numbers(istream_iterator<int>(lineStream),{});
        if (res.empty()) {
            for (auto number: numbers) {
                res.push_back(vector<int>{number});
            }
        }
        else {
            for (int i = 0; i < numbers.size(); ++i) {
                res[i].push_back(numbers[i]);
            }
        }
    }
    ifs.close();
    return res;
}

vector<vector<int> > read_hap_gz(const string &filename){
    vector<vector<int> > res;
    ifstream file(filename, ios_base::in | ios_base::binary);
    filtering_streambuf<input> inbuf;
    inbuf.push(gzip_decompressor());
    inbuf.push(file);
    //Convert streambuf to istream
    istream instream(&inbuf);
    //Iterate lines
    string line;
    while(getline(instream, line)) {
        stringstream lineStream(line);
        vector<int> numbers(istream_iterator<int>(lineStream), {});
        if (res.empty()) {
            for (auto number: numbers) {
                res.push_back(vector<int>{number});
            }
        } else {
            for (int i = 0; i < numbers.size(); ++i) {
                res[i].push_back(numbers[i]);
            }
        }
    }
    file.close();
    return res;
}