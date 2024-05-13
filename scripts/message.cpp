#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    const string myMessage = "You are the best!";
    cout << myMessage;
    cout << endl;
    printf("No, YOU!!!!!");
    cout << endl;

    vector<string> msg {"I", "love", "you,", "too"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
}