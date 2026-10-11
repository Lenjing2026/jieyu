#include<string>
using namespace std;
struct __consten_pool{
    struct version{
        string charv;
        int a,b,c;
    } version;
    string downloand_mj;
};
const __consten_pool consten_pool={
    .version={"0.2.2",0,2,2},
    .downloand_mj="null"
};