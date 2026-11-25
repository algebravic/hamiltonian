// C++ program by Bert Dobbelaere computing OEIS sequence A071984:
// "Square loops: the number of circular permutations (reversals not counted as different)
//  of the numbers 1 to n such that the sum of any two consecutive numbers is a square."

// Algorithm is based on recursive search.
// The numbers 1..N are considered vertices of a graph, connected by edges iff both sides add up to a square.
// A considerable speedup is achieved by guarding at each step that each unvisited vertex has at least two
// edges connecting it to other vertices than non-terminal vertices in the path composed so far (necessary
// condition to complete a Hamilton cycle)
// 
// Idea used to avoid mirroring the circular permutation:
// Number 50 has only two edges for N<=70 (14 and 31 ; 50+14 and 50+31 being the only squares possible), 
// so we can always start with "14 50 31" assuming 50<=N<=70
// Idem for "15 49 32" (N=49) and for "4 32 17" (32<=N<=48)
//
// Results:
// A071984(32) = 1
// A071984(33) = 1
// A071984(34) = 11
// A071984(35) = 57
// A071984(36) = 31
// A071984(37) = 20
// A071984(38) = 25
// A071984(39) = 50
// A071984(40) = 64
// A071984(41) = 464
// A071984(42) = 1062
// A071984(43) = 4337
// A071984(44) = 10091
// A071984(45) = 21931
// A071984(46) = 69623
// A071984(47) = 115913
// A071984(48) = 227893
// A071984(49) = 457707
// A071984(50) = 297126
// A071984(51) = 1051583
// A071984(52) = 3377189
// A071984(53) = 7618873
// A071984(54) = 12476654
// A071984(55) = 25832098
// A071984(56) = 55792448
// A071984(57) = 75126741
// A071984(58) = 129180538
// A071984(59) = 357114149
// A071984(60) = 823402071
// A071984(61) = 3902161448

#include <iostream>
#include <cstring>
#include <cassert>

#if __cplusplus < 201103L
// Types below work with GCC.
typedef unsigned long long u64;
typedef unsigned u32;
#else
// C++11 has standard types
#include <cstdint>
typedef uint64_t u64;
typedef uint32_t u32;
#endif

#define NMAX 70
#define MAX_EDGES 7

struct NodeInfo_t{
	u32 edge[MAX_EDGES];
	u32 nedges;
	bool isactive;
	bool isused;
};

// Globals
NodeInfo_t ctxt[NMAX+1];
bool isSquare[2*NMAX];
u32 N, V0, V1, V2;
u64 count;

/**
 * Init square table and build the graph.
 */
void init()
{
	for(int k=0;k<(2*NMAX);k++)
		isSquare[k]=false;
	for(int k=1;k*k<(2*NMAX);k++)
		isSquare[k*k]=true;
	memset(ctxt,0,sizeof(ctxt));
	for(u32 k=1;k<=N;k++)
	{
		for(u32 l=1;l<=N;l++)
		{
			if(l==k)
				continue;
			if(isSquare[k+l])
			{
				ctxt[k].edge[ctxt[k].nedges]=l;
				ctxt[k].nedges++;
				assert(ctxt[k].nedges<=MAX_EDGES);
			}
		}
		ctxt[k].isactive=true;
	}
}

/**
 * Recursively count the cycles
 * @param level Number of numbers already assigned
 * @param latest Value of last assigned number
 */
void solve(u32 level, u32 latest)
{
	if(level==N)
	{
		if(isSquare[latest+V0])
			count++;
	}
	else
	{
		NodeInfo_t &ls_latest=ctxt[latest];
		for(u32 t=0;t<ls_latest.nedges;t++)
		{
			u32 nexthop=ls_latest.edge[t];
			if(!ctxt[nexthop].isused)
			{
				ctxt[nexthop].isused=true;
				
				// All "unused" candidate neighbours of 'latest' leaving behind
				// should have >=2 "active" candidates left (including nexthop and V0)
				bool ok=true;
				for(u32 tt=0;tt<ls_latest.nedges;tt++)
					if(!ctxt[ls_latest.edge[tt]].isused)
					{
						u32 linkcnt=0;
						u32 node=ls_latest.edge[tt];
						for(u32 ttt=0;ttt<ctxt[node].nedges;ttt++)
							if(ctxt[ctxt[node].edge[ttt]].isactive)
							{
								linkcnt++;
							}
						if(linkcnt<2)
						{
							ok=false;
							break;
						}
					}
				
				if(ok)
				{
					ctxt[nexthop].isactive=false;
					solve(level+1, nexthop);
					ctxt[nexthop].isactive=true;
				}
				
				ctxt[nexthop].isused=false;	
			}
		}
	}
}

int main()
{
	for(N=32;N<=NMAX;N++)
	{
		// Determine initial triple. For N<=70, a 2nd element can be chosen that allows only two neighbours.
		// The relative order of the 1st and the 3rd element then fixes the direction of the cycle, so we
		// don't count duplicates.
		if(N<=48)
		{
			V0=4; V1=32; V2=17;
		}
		else if(N==49)
		{
			V0=15; V1=49; V2=32;
		}
		else if(N<=70)
		{
			V0=14; V1=50; V2=31;
		}
		else
		{
			// We need a different trick to avoid mirror counts if ever N>70 - not implemented.
			assert(false); 
		}
		
		// Initialize context and fill 1st three elements
		init();
		count=0;
		ctxt[V0].isused=true;
		ctxt[V1].isused=true;
		ctxt[V2].isused=true;
		ctxt[V1].isactive=false;
		ctxt[V2].isactive=false;
		// Count and print
		solve(3, V2);
		std::cout << "A071984(" << N << ") = " << count << std::endl;
	}
	return 0;
}
