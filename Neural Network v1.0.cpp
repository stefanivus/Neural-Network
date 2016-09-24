/** 
Neural Network algorithm
Copyright (C) 2016 Stefan Ivanovic 
This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>
**/

#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>

const double e = 2.71828182845904523536028747135266249775724709369995;

using namespace std;


class Neuron//Neuron class
{
	public:
	double a;
	double z;
	double a_prime;
	vector<double> weights;
	Neuron(int x);
	void Sigmoid(vector<double> x);
	
};

Neuron::Neuron(int x)
{
   for (int i=0;i<x;i++)//Gives weights to the neuron based on how many neurons the prev layer has
   {
   	    double gi = (double) rand()/(double)(RAND_MAX);
		weights.push_back(gi);
   }
}
void Neuron::Sigmoid(vector<double> x)//Sigmoid function 
{
	int i;
	for(i=0;i<x.size();i++)
	{
		z += x.at(i)*this->weights.at(i);
	}
	a = 1/(1+pow(e,-z+1));
}



class Net//Neural Network
{
	public:
	vector<double> inputs;//Inputs
	vector<double> outputs;//Desired outputs
	vector<double> hid_outputs;//Outputs from hidden layer,inputs to output layer
	vector<double> out_outputs;//Outputs from output layer
	vector<Neuron> hid_neurons;//Neurons in hidden layer
	vector<Neuron> out_neurons;//Neurons in output layer
	Net(vector<double> in, int hidLayer_size, vector<double> out);//Constructor
	void Forward_Prop();//Forward propagate
	void Update_weights();
	void train(int x, Net net1);
	void Save(vector<double> hid_weights,vector<double> out_weights);
};

Net::Net(vector<double> in, int hidLayer_size, vector<double> out)
{   //Constructor stores inputs,output,initial weights,Neurons into vectors
	int i;
	for (i=0;i<hidLayer_size;i++)
	{
		Neuron N(in.size());
		hid_neurons.push_back(N);
	}
	for (i=0;i<out.size();i++)
	{
		Neuron N(hid_neurons.size());
		out_neurons.push_back(N);
	}
	for(double i : in)
	{
		inputs.push_back(i);
	}
	for(double i : out)
	{
		outputs.push_back(i);
	}
}

void Net::Forward_Prop()//Forward Propagate
{
	int i;
	hid_outputs.clear();
	out_outputs.clear();
	for(Neuron j : hid_neurons)
	{
		j.Sigmoid(inputs);
		hid_outputs.push_back(j.a);
	}
	for(Neuron k : out_neurons)
	{
		k.Sigmoid(hid_outputs);
		out_outputs.push_back(k.a);
	}
}

void Net::Update_weights()
{
	double val = 0;
	for(Neuron &o : out_neurons)
	{
	   int i;
	   int count= 0;
	  for(i=0;i<o.weights.size();i++)
	   {
		   o.weights.at(i) = o.weights.at(i) - (out_outputs.at(count)-outputs.at(count))*out_outputs.at(count)*(1-out_outputs.at(count))*hid_outputs.at(count);
		   val += (out_outputs.at(count)-outputs.at(count))*out_outputs.at(count)*(1-out_outputs.at(count))*o.weights.at(i);
	   }
	   count++;
	}
	for(Neuron &h : hid_neurons)
	{
	   int i;
	   int count = 0;
	  for(i=0;i<h.weights.size();i++)
	   {
		   h.weights.at(i) -= val*hid_outputs.at(count)*(1-hid_outputs.at(count))*inputs.at(i);
	   }
	   count++;
	}
}


void Net::Save(vector<double> hid_weights,vector<double> out_weights)
{
	string name,full_name_h,full_name_o;
	ofstream file1;
	ofstream file2;
	cout << "Enter name of save file:";
	cin >> name;
	full_name_h = name + "_h.txt";
	file1.open(full_name_h, ios_base::out);
	for(double i : hid_weights)
	{
		file1 << i << "\n";
	}
	file1.close();
	full_name_o = name + "_o.txt";
	file2.open(full_name_o, ios_base::out);
	for(double i : out_weights)
	{
		file2 << i << "\n";
	}
	file2.close();
	
}

void train(int x, Net &Net1)
{
	int i;
	vector<double> hid_weights;
	vector<double> out_weights;
	cout << "Training in progress..."<< endl;
	for(i=0;i<x;i++)
	{
		
		Net1.Forward_Prop();
        Net1.Update_weights();
	}
	for(Neuron &h : Net1.hid_neurons)
	{
		for(i=0;i<h.weights.size();i++)
		{
			hid_weights.push_back(h.weights[i]);
		}
	}
	for(Neuron &o : Net1.out_neurons)
	{
		for(i=0;i<o.weights.size();i++)
		{
			out_weights.push_back(o.weights[i]);
		}
	}
	cout << "Training finished!"<< endl;
	cout << "Result output is: " << Net1.out_outputs[0]*100 << endl;
	Net1.Save(hid_weights,out_weights);
}

void Load(string filename)
{
	int i,x;
	int j = 0;
	string line,filename_h,filename_o;
	vector<double> in;
	vector<double> out;
	out.push_back(1);
	vector<double> hid_weights;
	vector<double> out_weights;
	filename_h = filename + "_h.txt";
	filename_o = filename + "_o.txt";
	ifstream file1(filename_h);
	ifstream file2(filename_o);
	
	while ( getline (file1,line) )
    {
       hid_weights.push_back(stod(line));
    }
    while ( getline (file2,line) )
    {
       out_weights.push_back(stod(line));
    }
    
    cout << "Would you like to train your net further or just run it once?(1=train,2=run):";
    int choice;
    cin >> choice;
    for(i=0;i<hid_weights.size()/4;i++)
    {
       cout << "Enter input:";
       cin >> x;
       in.push_back(x);
    }
    if (choice == 1)
    {
        int des_out;
        cout <<"Enter desired output:";
        cin >> des_out;
        out.clear();
        out.push_back(des_out);
        Net Net1(in,4,out);
        for(Neuron &h : Net1.hid_neurons)
        {
    	    for(i=0;i<h.weights.size();i++)
    	    {
    		    h.weights[i] = hid_weights[j];
    		    j++;
		    }
	    }
	    j=0;
	    for(Neuron &o : Net1.out_neurons)
        {
    	    for(i=0;i<o.weights.size();i++)
    	    {
    		    o.weights[i] = out_weights[j];
    		    j++;
		    }
	    }
	    int t;
	    cout << "Enter training time:";
	    cin >> t;
	    train(t*10000,Net1);
    } 
    
   else if(choice == 2)
   {
    Net Net1(in,4,out);
    for(Neuron &h : Net1.hid_neurons)
    {
    	for(i=0;i<h.weights.size();i++)
    	{
    		h.weights[i] = hid_weights[j];
    		j++;
		}
	}
	j=0;
	for(Neuron &o : Net1.out_neurons)
    {
    	for(i=0;i<o.weights.size();i++)
    	{
    		o.weights[i] = out_weights[j];
    		j++;
		}
	}
	
	Net1.Forward_Prop();
	for(double k : Net1.out_outputs)
	{
		cout << k*100 << endl;
	}
   }
}

int main()
{
	srand (time(NULL));
	int choice;
	cout << "                                Welcome to Neural Network v1.0!" << endl;
	cout << "Would you like to train a network or load an existing one?(1=train,2=load)";
	cin >> choice;
if (choice == 1)
{
	int i,num,time;
	double in,out;
	vector<double> inputs;
	vector<double> outputs;
	cout << "Enter number of inputs:";
	cin >> num;
	for(i=0;i<num;i++)
	{
		cout << "Enter input:";
		cin >> in;
		cout << "\n";
		inputs.push_back(in);
	}
	
	cout << "Enter desired output:";
	cin >> out;
	outputs.push_back(out/100);
	cout << "Enter desired training time:";
	cin >> time;
	

	


	
	Net Net1(inputs,4,outputs);//Define net
	train(time*10000,Net1);
	
}

else
{
	string file;
	cout << "Enter name of the file you want to load:";
	cin >> file;
	Load(file);
}
	
	
	
	
	
	
	return 0;
}
