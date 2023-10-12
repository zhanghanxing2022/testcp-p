#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
using namespace std;
const float pi = 3.14159265359;
// g++ -o MyProgram Layer.cpp  -Wall -std=c++17 -I/opt/homebrew/Cellar/sfml/2.6.0/include/ -L/opt/homebrew/Cellar/sfml/2.6.0/lib/ -lsfml-graphics -lsfml-window -lsfml-system&&./MyProgram
#define myLog() cout << __FILE__ << " " << __LINE__ << " " << __FUNCTION__ << endl;
void myPrint(const vector<vector<float>> &res)
{
    for (int i = 0; i < res.size(); i++)
    {
        for (int j = 0; j < res[i].size(); j++)
        {
            printf("%.5f ,%x", res[i][j], &(res[i][j]));
        }
        cout << endl;
    }
}
void myPrint(const vector<vector<float>> &res, ofstream &out)
{
    for (int i = 0; i < res.size(); i++)
    {
        for (int j = 0; j < res[i].size(); j++)
        {
            out << setprecision(20) << res[i][j] << " ";
        }
        out << endl;
    }
}
vector<vector<float>> matrix(int a, int b, float x = 0)
{
    vector<float> line(b, x);
    vector<vector<float>> map(a, line);
    return map;
}
vector<vector<float>> matrixRandom(int a, int b)
{
    random_device rd;                       // 用于生成随机数种子
    mt19937 r_eng(rd());                    // 随机数生成器
    normal_distribution<float> dis(1, 1.5); // 随机数分布器，均值、方差
    vector<float> line(b, 0);
    vector<vector<float>> map(a, line);
    for (auto &x : map)
    {
        for (auto &y : x)
        {
            y = dis(r_eng);
        }
    }
    return map;
}
void matrixDiv(vector<vector<float>> &map, float r_eng)
{

    for (auto &x : map)
    {
        for (auto &y : x)
        {
            y /= sqrt(r_eng);
        }
    }
}
vector<vector<float>> as(const vector<vector<float>> &x, float a = 0)
{
    vector<float> line(x[0].size(), a);
    vector<vector<float>> map(x.size(), line);
    return map;
}
vector<vector<float>> dot(const vector<vector<float>> &left, const vector<vector<float>> &right)
{
    int k, m, n;
    k = left.size();
    m = left[0].size();
    if (right.size() != m)
    {
        assert(false);
    }
    n = right[0].size();
    vector<float> line(n, 0);
    vector<vector<float>> map(k, line);
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float res = 0.0;
            for (int k = 0; k < m; k++)
            {
                res += left[i][k] * right[k][j];
            }
            map[i][j] = res;
        }
    }
    return map;
}
vector<vector<float>> T(vector<vector<float>> &left)
{
    auto res = matrix(left[0].size(), left.size());
    for (int i = 0; i < left.size(); i++)
    {
        for (int j = 0; j < left[0].size(); j++)
        {
            res[j][i] = left[i][j];
        }
    }
    return res;
}
vector<vector<float>> add(const vector<vector<float>> &left, const vector<vector<float>> &right)
{

    if (left.size() != right.size() || left[0].size() != right[0].size())
    {
        assert(false);
    }
    auto res = as(left);
    for (int i = 0; i < res.size(); i++)
    {
        for (int j = 0; j < res[0].size(); j++)
        {
            res[i][j] = left[i][j] + right[i][j];
        }
    }
    return res;
}
class Activator
{
public:
    virtual vector<vector<float>> call(const vector<vector<float>> &) = 0;
    virtual vector<vector<float>> derivation(const vector<vector<float>> &) = 0;
};
class Sigmoid : public Activator
{
public:
    vector<vector<float>> call(const vector<vector<float>> &x)
    {
        auto res = as(x);
        for (int i = 0; i < x.size(); i++)
        {

            for (int j = 0; j < x[0].size(); j++)
            {
                res[i][j] = 1 / (1 + exp(-x[i][j]));
            }
        }
        return res;
    }
    vector<vector<float>> derivation(const vector<vector<float>> &res)
    {
        auto det = matrix(res.size(), res.size());
        for (int i = 0; i < res.size(); i++)
        {
            det[i][i] = (1 - res[i][0]) * res[i][0];
        }
        return det;
    }
};
class X : public Activator
{
public:
    vector<vector<float>> call(const vector<vector<float>> &x)
    {

        return x;
    }
    vector<vector<float>> derivation(const vector<vector<float>> &res)
    {
        auto det = matrix(res.size(), res.size());
        for (int i = 0; i < res.size(); i++)
        {
            det[i][i] = 1;
        }
        return det;
    }
};
class Layer
{
public:
    mutable vector<vector<float>> input;
    mutable vector<vector<float>> weights;
    mutable vector<vector<float>> bias;
    mutable vector<vector<float>> result;

    mutable vector<vector<float>> act_div_res;
    mutable vector<vector<float>> loss_div_bias;
    mutable vector<vector<float>> loss_div_weights;
    mutable vector<vector<float>> backPropagationB;
    mutable vector<vector<float>> backPropagationW;

    Activator *act;
    int Idim;
    int Odim;
    Layer(int inputs, int neurons, Activator *act0)
    {
        Idim = inputs;
        Odim = neurons;
        input = matrix(inputs, 1);
        this->weights = matrixRandom(neurons, inputs);
        bias = matrixRandom(neurons, 1);
        result = as(bias);

        act_div_res = matrix(neurons, neurons);
        loss_div_bias = matrix(1, neurons);
        loss_div_weights = as(this->weights);
        backPropagationB = as(bias);
        backPropagationW = as(this->weights);
        matrixDiv(weights, neurons);
        matrixDiv(bias, neurons);
        act = act0;
    }
    vector<vector<float>> forward(vector<vector<float>> &x)
    {
        input = x;
        this->result = dot(this->weights, input);
        this->result = add(this->result, bias);
        this->result = act->call(this->result);
        act_div_res = act->derivation(this->result);
        return this->result;
    }
    volatile vector<vector<float>> feedBackward(vector<vector<float>> &feedback)
    {

        loss_div_bias = dot(feedback, act_div_res);
        loss_div_weights = dot(T(loss_div_bias), T(input));
        backPropagationW = add(backPropagationW, loss_div_weights);
        backPropagationB = add(backPropagationB, T(loss_div_bias));

        // cout << "Feed" << endl;
        // myPrint(feedback);
        // cout << "act_div_res" << endl;
        // myPrint(act_div_res);
        // cout << "loss_div_bias:" << endl;

        // myPrint(loss_div_bias);
        // cout << "result" << endl;
        // myPrint(result);
        return dot(loss_div_bias, this->weights);
    }
    volatile void parameterUpdate(int sum, float lr)
    {

        for (int i = 0; i < Odim; i++)
        {
            for (int j = 0; j < Idim; j++)
            {
                // cout<<weights[i][j]<<":";
                this->weights[i][j] += backPropagationW[i][j] * lr / sum;
                // cout<<weights[i][j]<<endl;
            }
        }
        for (auto &x : backPropagationW)
        {
            for (auto &y : x)
            {
                y = 0;
            }
        }
        for (int i = 0; i < Odim; i++)
        {
            bias[i][0] += backPropagationB[i][0] * lr / sum;
        }
        for (auto &x : backPropagationB)
        {
            for (auto &y : x)
            {
                y = 0;
            }
        }
    }
};
class LossFunc
{
public:
    virtual vector<vector<float>> callLoss(vector<vector<float>> &f, vector<vector<float>> &res) = 0;
    virtual vector<vector<float>> callDerivation(vector<vector<float>> &f, vector<vector<float>> &res) = 0;
};
class Square_Difference : public LossFunc
{
public:
    vector<vector<float>> callLoss(vector<vector<float>> &f, vector<vector<float>> &res)
    {
        auto loss = as(f);
        for (int i = 0; i < f.size(); i++)
        {
            for (int j = 0; j < f[0].size(); j++)
            {
                loss[i][j] = (f[i][j] - res[i][j]) * (f[i][j] - res[i][j]);
            }
        }
        return loss;
    }
    vector<vector<float>> callDerivation(vector<vector<float>> &f, vector<vector<float>> &res)
    {
        auto Derivation = as(f);
        for (int i = 0; i < f.size(); i++)
        {
            for (int j = 0; j < f[0].size(); j++)
            {
                Derivation[i][j] = 2 * (f[i][j] - res[i][j]);
            }
        }
        return Derivation;
    }
};
class Network
{
public:
    LossFunc *loss;
    vector<Layer> network;
    Network(LossFunc &loss0)
    {
        loss = &loss0;
    }
    void addLayer(Layer layer)
    {
        network.push_back(layer);
    }
    vector<vector<float>> predict(vector<vector<float>> x)
    {
        vector<vector<float>> res = x;
        for (auto &layer : network)
        {

            auto out = layer.forward(res);
            res = out;
        }
        return res;
    }
    void log(string output)
    {
        ofstream ofs(output, std::ios::out);

        for (auto &layer : network)
        {
            ofs << layer.Idim << " " << layer.Odim << " " << endl;
            ofs << "weights:" << endl;
            myPrint(layer.weights, ofs);
            ofs << "bias:" << endl;
            myPrint(layer.bias, ofs);
        }
        ofs.close();
    }
    void back(vector<vector<float>> &f, vector<vector<float>> &res)
    {
        auto loss0 = loss->callLoss(f, res);
        auto det = loss->callDerivation(f, res);

        for (int i = network.size() - 1; i >= 0; i--)
        {
            auto res = network[i].feedBackward(det);
            det = res;
        }
    }
    volatile void update(int sum, float lr)
    {
        for (auto &x : network)
        {
            x.parameterUpdate(sum, lr);
        }
    }
};
int train(string filename)
{
    Square_Difference lossF = Square_Difference();
    Network model(lossF);
    model.addLayer(Layer(1, 10, new Sigmoid()));
    model.addLayer(Layer(10, 10, new Sigmoid()));
    model.addLayer(Layer(10, 1, new X()));

    vector<float> x;
    vector<float> y;
    int size = 5000;
    for (int i = 0; i < size; i++)
    {
        x.push_back(-M_PI + ((2 * M_PI) / (size - 1)) * i);
        y.push_back(sin(x.back()));
        // cout << x.back() << " " << y.back() << endl;
    }
    int epoch = 500;
    float lr = 0.1;
    int batch_size = 2;
    std::vector<int> l(size, 0);
    std::iota(l.begin(), l.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd()); // 随机数引擎
    float sumLoss = 0;
    float tempLoss = 0;
    int worse = 0;
    float all_loss = 0;
    int iteration = size / batch_size;
    for (int i = 0; i < epoch; i++)
    {
        shuffle(l.begin(), l.end(), g);
        tempLoss = sumLoss;
        sumLoss = 0;

        for (int j = 0; j < iteration; j++)
        {
            all_loss = 0;
            for (int k = 0; k < batch_size; k++)
            {
                auto X = matrix(1, 1);
                X[0][0] = x[l[j * batch_size + k]];

                auto Y = matrix(1, 1);
                Y[0][0] = y[l[j * batch_size + k]];
                auto res = model.predict(X);
                auto loss_i = model.loss->callLoss(Y, res);
                all_loss += loss_i[0][0];
                // cout<<loss_i[0][0]<<" "<<x[j*batch_size+k]<<" "<<y[j*batch_size+k]<<" "<<res[0][0]<<endl;
                model.back(Y, res);
            }
            sumLoss += all_loss / batch_size;
            model.update(batch_size, lr);
        }

        if (sumLoss * 1.01 > tempLoss)
        {
            worse++;
        }
        if (worse >= 5)
        {
            lr = max(lr * 0.9, 0.0001);
            worse = 0;
        }
    }
    printf("epoch:500,loss:%.20f,iteration:%d\n",sumLoss / iteration, iteration);
    vector<float> xx;
    vector<float> yy;
    vector<float> yyy;

    float maxf = 0.0;
    float meanf = 0.0;
    for (int i = 0; i < 1000; i++)
    {
        xx.push_back(-M_PI + ((2 * M_PI) / (1000 - 1)) * i);
        auto temp = matrix(1, 1);
        temp[0][0] = xx.back();
        yy.push_back(model.predict(temp)[0][0]);
        yyy.push_back(sin(xx.back()));
        meanf += fabs(yy.back() - yyy.back());
        maxf = max(maxf, fabs(yy.back() - yyy.back()));
    }
    printf("maxf:%f,meanf:%f", maxf, meanf / 1000);
    model.log(filename);
}
int main()
{
    for (int i = 0; i < 1000; i++)
    {
        train(to_string(i) + ".txt");
    }
    
    return 0;
}