#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    const string myMessage = "taka";
    cout << myMessage;
    cout << endl;
    printf("this is another message");
    cout << endl;

    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
}