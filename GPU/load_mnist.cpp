#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

vector<float> operator-(const vector<float> &m1, const float m2) {
    const unsigned long VECTOR_SIZE = m1.size();
    vector<float> difference(VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
        difference[i] = m1[i] - m2;
    };

    return difference;
}

vector<float> operator/(const vector<float> &m2, const float m1) {
    const unsigned long VECTOR_SIZE = m2.size();
    vector<float> product(VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
        product[i] = m2[i] / m1;
    };
    return product;
}

void load_mnist(vector<float> &x_train, vector<float> &y_train, int *data_size) {
    string line;
    vector<string> line_v;

    cout << "Loading data ...\n";

    ifstream myfile("train.txt");
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(), 0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit) {
                    y_train.push_back(1.);
                } else {
                    y_train.push_back(0.);
                }
            }
            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i) {
                x_train.push_back(strtof((line_v[i]).c_str(), 0));
            }
        }
        x_train = x_train / 255.0;
        x_train = x_train - 0.1307;
        x_train = x_train / 0.3081;
        myfile.close();
    }
    *data_size = y_train.size() / 10;
}