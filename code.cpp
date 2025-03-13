ICPC Amritapuri Regionals 2024-2025
Team Koders Klub
Walchand College Of  Engineering, Sangli
Reference Material
#define MAXN 100005
const int MOD = 1e9 + 7;
Fast IO : 
ios_base::sync_with_stdio(false);
cin.tie(NULL);
#pragma GCC target("popcnt")
Fenwick Tree : 
struct fenwick {
    vector<int> fn;
    int n;
    void init(int n) {
        this->n = n + 1;
        fn.resize(this->n, 0);
    }
    void add(int x, int y) {
        x++;
        while (x < n) {
            fn[x] += y;
            x += (x & (-x));
        }
    }
    int sum(int x) {
        x++;
        int ans = 0;
        while (x) {
            ans += fn[x];
            x -= (x & (-x));
        }
        return ans;
    }
    int sum(int l, int r) {
        return sum(r) - sum(l - 1);
    }
};
Sparse Table : 
#define K 17
long long st[K + 1][MAXN];
void buildSparseTable(vector<long long>& array, int N) {
    for (int i = 0; i < N; i++) st[0][i] = array[i];
    for (int i = 1; i <= K; i++)
        for (int j = 0; j + (1 << i) <= N; j++)
            st[i][j] = st[i - 1][j] + st[i - 1][j + (1 << (i - 1))];
}

long long query(int L, int R) {
    long long sum = 0;
    for (int i = K; i >= 0; i--)
        if ((1 << i) <= R - L + 1) {
            sum += st[i][L];
            L += 1 << i;
        }
    return sum;
}

Lazy Segment Tree
template<typename Node, typename Update>
struct LazySGT {
    vector<Node> tree;
    vector<bool> lazy;
    vector<Update> updates;
    vector<ll> arr; // type may change
    int n,s;
    LazySGT(int a_len, vector<ll> &a) { // change if type updated
        arr = a; n = a_len; s = 1;
        while(s < 2 * n)  s = s << 1;
        tree.resize(s); fill(all(tree), Node());
        lazy.resize(s); fill(all(lazy), false);
        updates.resize(s); fill(all(updates), Update());
        build(0, n - 1, 1);
    }
    void build(int start, int end, int index) { // Never change this
        if (start == end)   {
            tree[index] = Node(arr[start]);
            return;
        }
        int mid = (start + end) / 2;
        build(start, mid, 2 * index); build(mid + 1, end, 2 * index + 1);
        tree[index].merge(tree[2 * index], tree[2 * index + 1]);
    }
    void pushdown(int index, int start, int end){
        if(lazy[index]){
            int mid = (start + end) / 2;
            apply(2 * index, start, mid, updates[index]);
apply(2 * index + 1, mid + 1, end, updates[index]);
            updates[index] = Update();
            lazy[index] = 0;
        }
    }
    void apply(int index, int start, int end, Update& u){
        if(start != end){
            lazy[index] = 1;
            updates[index].combine(u, start, end);
        }
        u.apply(tree[index], start, end);
    }
    void update(int start, int end, int index, int left, int right, Update& u) { //Never Change this
        if(start > right || end < left) return;
        if(start >= left && end <= right){
            apply(index, start, end, u);
            return;
        }
        pushdown(index, start, end);
        int mid = (start + end) / 2;
        update(start, mid, 2 * index, left, right, u);
        update(mid + 1, end, 2 * index + 1, left, right, u);
        tree[index].merge(tree[2 * index], tree[2 * index + 1]);
    }
    Node query(int start, int end, int index, int left, int right) { // Never change this
        if (start > right || end < left) return Node();
        if (start >= left && end <= right){
            pushdown(index, start, end);
            return tree[index];
        }
        pushdown(index, start, end);
        int mid = (start + end) / 2;
        Node l, r, ans;
        l = query(start, mid, 2 * index, left, right);
        r = query(mid + 1, end, 2 * index + 1, left, right);
        ans.merge(l, r);
        return ans;    }
    void make_update(int left, int right, ll val) {  // pass in as many parameters as required
        Update new_update = Update(val); // may change
        update(0, n - 1, 1, left, right, new_update);
    }
    Node make_query(int left, int right){ return query(0, n - 1, 1, left, right); }    
};
struct Node1 {
    ll val; // may change
    Node1() { // Identity element
        val = 0;    // may change
    }
    Node1(ll p1) {  // Actual Node
        val = p1; // may change
    }
    void merge(Node1 &l, Node1 &r) { // Merge two child nodes
        val = l.val + r.val;  // may change
    }
};
struct Update1 {
    ll val; // may change
    Update1(){ // Identity update
        val = 0;
    }
    Update1(ll val1) { // Actual Update
        val = val1;
    }
    void apply(Node1 &a, int start, int end) { // apply update to given node
        a.val = val * (end - start + 1); // may change
    }
    void combine(Update1& new_update, int start, int end){
        val = new_update.val;
    }
};
Sparse Segment Tree : 
class SparseSegtree {
  private:
	struct Node {
		int freq = 0, lazy = 0;
		Node *left = nullptr, *right = nullptr;
	};
	Node *root = new Node;
	const int n;
	int comb(int a, int b) { return a + b; }
	void apply(Node *cur, int len, int val) {
		if (val == 1) {
			(cur->lazy) = val;
			(cur->freq) = len * val;
		}
	}
	void push_down(Node *cur, int l, int r) {
		if ((cur->left) == nullptr) { (cur->left) = new Node; }
		if ((cur->right) == nullptr) { (cur->right) = new Node; }
		int m = (l + r) / 2;
		apply(cur->left, m - l + 1, cur->lazy);
		apply(cur->right, r - m, cur->lazy);
	}
	void range_set(Node *cur, int l, int r, int ql, int qr, int val) {
		if (qr < l || ql > r) { return; }
		if (ql <= l && r <= qr) apply(cur, r - l + 1, val);
		else {
			push_down(cur, l, r);
			int m = (l + r) / 2;
			range_set(cur->left, l, m, ql, qr, val);
			range_set(cur->right, m + 1, r, ql, qr, val);
			(cur->freq) = comb((cur->left)->freq, (cur->right)->freq);
		}
	}
	int range_sum(Node *cur, int l, int r, int ql, int qr) {
		if (qr < l || ql > r) { return 0; }
		if (ql <= l && r <= qr) { return cur->freq; }
		push_down(cur, l, r);
		int m = (l + r) / 2;
		return comb(range_sum(cur->left, l, m, ql, qr),
		            range_sum(cur->right, m + 1, r, ql, qr));
	}
  public:
	SparseSegtree(int n) : n(n) {}
	void range_set(int ql, int qr, int val) { range_set(root, 0, n - 1, ql, qr, val); }
	int range_sum(int ql, int qr) { return range_sum(root, 0, n - 1, ql, qr); }
};
	SparseSegtree st(1e9 + 1);
		int c = 0;
		if (type == 1) {
			c = st.range_sum(x + c, y + c);
			cout << c << '\n';
		} else if (type == 2) {
			st.range_set(x + c, y + c, 1);
		}
Disjoint Set (DSU) : 
class DisjointSet {
    vector<int> parent, size;
public:
    DisjointSet(int n) {
        parent.resize(n + 1);
        size.resize(n + 1);
        for (int i = 0; i <= n; i++) {
            parent[i] = i; size[i] = 1;
        }
    }
    int findUPar(int node) {
        if (node == parent[node]) return node;
        return parent[node] = findUPar(parent[node]);
    }
    void unionBySize(int u, int v) {
        int ulp_u = findUPar(u), ulp_v = findUPar(v);
        if (ulp_u == ulp_v) return;
        if (size[ulp_u] < size[ulp_v]) {
            parent[ulp_u] = ulp_v;
            size[ulp_v] += size[ulp_u];
        }
        else {
            parent[ulp_v] = ulp_u;
            size[ulp_u] += size[ulp_v];
        }
    }
};
ModInt Class : 
template <i64 mod = MOD>
struct ModInt{
    i64 p;
    ModInt() : p(0){}
    ModInt(i64 x){p = x >= 0 ? x % mod : x + (-x + mod - 1) / mod * mod;}
    ModInt& operator+=(const ModInt& y){p = p + *y - ((p + *y) >= mod ? mod : 0); return *this;}
    ModInt& operator-=(const ModInt& y){p = p - *y + (p - *y < 0 ? mod : 0); return *this;}
    ModInt& operator*=(const ModInt& y){p = (p * *y) % mod; return *this;}
    ModInt& operator%=(const ModInt& y){if(y)p %= *y; return *this;}
    ModInt operator+(const ModInt& y) const{ModInt x = *this; return x += y;}
    ModInt operator-(const ModInt& y) const{ModInt x = *this; return x -= y;}
    ModInt operator*(const ModInt& y) const{ModInt x = *this; return x *= y;}
    ModInt operator%(const ModInt& y) const{ModInt x = *this; return x %= y;}
    friend ostream& operator<<(ostream& stream, const ModInt<mod>& x){
        stream << *x;
        return stream;
    }
    friend ostream& operator>>(ostream& stream, const ModInt<mod>& x){
        stream >> *x;
        return stream;
    }
    ModInt& operator++(){p = (p + 1) % mod; return *this;}
    ModInt& operator--(){p = (p - 1 + mod) % mod; return *this;}
    bool operator==(const ModInt& y) const{return p == *y;}
    bool operator!=(const ModInt& y) const{return p != *y;}
    const i64& operator*() const{return p;}
    i64& operator*(){return p;}
};
using mint = ModInt<>;
Power Function : 
ll gcd(ll a, ll b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}
ll binpow(ll a,ll b){
    ll ans = 1;
    while(b > 0){
        if((b & 1) == 1) {
        	ans *= a;
        	ans %= mod;
        }
        a *= a;
        a %= mod;
        b = b >> 1;
    }
    return ans;
}
ll inv(ll a) {
    return binpowmod(a, mod - 2);
}
// Factorial
const ll N = 1e5 + 5;
ll f_[N], _f[N];
void _fp() {
    f_[0] = f_[1] = 1;
    for (int i = 2; i < N; i++) {
        f_[i] = f_[i - 1] * i;
        f_[i] %= mod;
    }
    _f[N - 1] = inv(f_[N - 1]);
    for (int i = N - 2; i >= 0; i--) {
        _f[i] = _f[i + 1] * (i + 1);
        _f[i] %= mod;
    }
}
ll nCr(ll x, ll y) {
    if (y > x) return 0ll;
    ll _n = f_[x];
    _n *= _f[y];
    _n %= mod;
    _n *= _f[x - y];
    _n %= mod
return _n;
}

Segmented Sieve : 
vector<bool> segmentedSieve(ll L, ll R) {
    ll lim = sqrt(R);
    vector<bool> mark(lim + 1, false);
    vector<ll> primes;
    for(ll i = 2; i <= lim; ++i) {
        if(!mark[i]) {
            primes.emplace_back(i);
            for(ll j = i * i; j <= lim; j += i) mark[j] = true;
        }
    }
    vector<bool> isPrime(R - L + 1, true);
    for(ll i : primes)
        for(ll j = max(i * i, (L + i - 1) / i * i); j <= R; j += i)
            isPrime[j - L] = false;
    if(L == 1) isPrime[0] = false;
    return isPrime;
}
Rabin-Karp
const int BASE = 31;
vector<long long> hash, revHash, power;
using string = vector<int>;    // for hashing vectors instead of strings
void preprocessHash(string &str) {
    int n = str.size();
    hash.resize(n + 1, 0);
    revHash.resize(n + 1, 0);
    power.resize(n + 1, 1);
    for(int i = 1; i <= n; ++i) {
        hash[i] = (hash[i - 1] * BASE + str[i - 1]) % MOD;
        revHash[i] = (revHash[i - 1] * BASE + str[n - i]) % MOD;
        power[i] = (power[i - 1] * BASE) % MOD;
    }
}
long long getHash(int l, int r) {
    long long result = (hash[r + 1] - hash[l] * power[r - l + 1] % MOD + MOD) % MOD;
    return result;
}
long long getRevHash(int l, int r, int n) {
    long long result = (revHash[n - l] - revHash[n - r - 1] * power[r - l + 1] % MOD + MOD) % MOD;
    return result;
}
bool isPali(string &str, int l, int r) {
    int n = str.size();
    return getHash(l, r) == getRevHash(l, r, n);
}
bool comparePref(int l1, int l2, int len) {
    return getHash(l1, l1 + len - 1) == getHash(l2, l2 + len - 1);
}
KMP
vector<int> prefix_function(string s) {
    int n = (int)s.length();
    vector<int> pi(n);
    for (int i = 1; i < n; i++) {
        int j = pi[i-1];
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];
        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi;
}
Z Function        
vector<int> z_function(string s) {
    int n = s.size();
    vector<int> z(n);
    int l = 0, r = 0;
    for(int i = 1; i < n; i++) {
        if(i < r) {
            z[i] = min(r - i, z[i - l]);
        }
        while(i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        if(i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }
    return z;
}
Xor of 1 to n:
static int findXOR(int n){
        int mod = n % 4;
        if (mod == 0) return n;
        else if (mod == 1) return 1;
        else if (mod == 2)  return n + 1;
        else if (mod == 3) return 0;
        return 0;
}
PBDS : 
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <ext/pb_ds/detail/standard_policies.hpp>
#include <functional>
using namespace __gnu_pbds;
typedef tree<int, null_type, less_equal<int>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
// ordered_set st; (use less_equal for multiset else use less for set)
// st.order_of_key(k); Returns the number of elements strictly smaller than k.
// st.find_by_order(k); Returns the address of the element at kth index in the set while using zero-based indexing
LCA Tree : 
const int MAXLEVELS = 20;
int up[MAXN + 1][MAXLEVELS]; 
int depth[MAXN + 1];         
void binary_lifting(int src, int par, vector<vector<int>>& adj) {
    up[src][0] = par;  
    for (int i = 1; i < MAXLEVELS; i++) {
        if (up[src][i - 1] != -1)    up[src][i] = up[up[src][i - 1]][i - 1];
        else    up[src][i] = -1;
    }
    for (auto child : adj[src]) {
        if (child != par) {
            depth[child] = depth[src] + 1;
            binary_lifting(child, src, adj);
        }
    }
}
int kLevelsAbove(int src, int k) {
    if (src == -1 || k == 0) return src;
    for (int i = MAXLEVELS - 1; i >= 0; i--) {
        if (k >= (1 << i)) {
            src = up[src][i];
            if (src == -1) break; 
            k -= (1 << i);
        }
    }
    return src;
}

int getLCA(int a, int b) {
    if (depth[a] > depth[b])    a = kLevelsAbove(a, depth[a] - depth[b]);
    else if (depth[b] > depth[a])   b = kLevelsAbove(b, depth[b] - depth[a]);
    if(a == b) return a; 
   for (int i = MAXLEVELS - 1; i >= 0; i--) {
        if (up[a][i] != up[b][i]) {
            a = up[a][i], b = up[b][i];
        }
    }
     return up[a][0];  
}

Dijkstra's Algo : 
const INF = 1e16;
void dijkstra(vector<vector<pair<int, int>>>& adj, int n) {
    vector<vector<ll>> dist(n + 1, vector<ll>(4, INF));
    for (int i = 1; i <= n; i++) {
        dist[i][0] = INF;  // MIN Dist 
        dist[i][1] = 1;    // Total number of paths
        dist[i][2] = -INF; // Max number of nodes till i
        dist[i][3] = INF;  // Min number of nodes till i
    }

    dist[1][0] = 0;    dist[1][1] = 1;    dist[1][2] = 0;    dist[1][3] = 0;    
    priority_queue<tuple<ll, int>, vector<tuple<ll, int>>, greater<tuple<ll, int>>> pq;
    pq.push({0, 1});

    while (!pq.empty()) {
        auto frontNode = pq.top();
        pq.pop();
        ll currentDist = get<0>(frontNode);
        int node = get<1>(frontNode);
        if (currentDist > dist[node][0]) continue;
        for (auto child : adj[node]) {
            int v, w;
            tie(v, w) = child;
            if (dist[node][0] + w < dist[v][0]) {
                dist[v][0] = dist[node][0] + w;
                pq.push({dist[v][0], v});

                dist[v][1] = dist[node][1];           // Paths count from parent
                dist[v][2] = dist[node][2] + 1;       // Max nodes in path
                dist[v][3] = dist[node][3] + 1;       // Min nodes in path
            } else if (dist[node][0] + w == dist[v][0]) {
                dist[v][1] = (dist[v][1] + dist[node][1])%MOD;          
                dist[v][2] = max(dist[v][2], dist[node][2] + 1); 
                dist[v][3] = min(dist[v][3], dist[node][3] + 1); 
            }
        }
    }
}

Manacher Algo : 
template<typename T>
vector<T> manacher_odd(const string& s) {
    int n = s.size();
    string t = "$" + s + "^";
    vector<T> p(n + 2);
    int l = 1, r = 1;
    for (int i = 1; i <= n; i++) {
        p[i] = max(0, min(r - i, p[l + (r - i)]));
        while (t[i - p[i]] == t[i + p[i]]) p[i]++;
        if (i + p[i] > r) l = i - p[i], r = i + p[i];
    }
    return vector<T>(begin(p) + 1, end(p) - 1);
}
vector<int> pal_lengths = manacher_odd<int>(s); 
for (int radius : pal_lengths) cout << 2 * radius - 1 << " "; // Full palindrome length 

Tarjan’s Algo : 
class Solution {
private:
    int timer = 1;
    void dfs(int node, int parent, vector<int> &vis,
             vector<int> adj[], int tin[], int low[], vector<vector<int>> &bridges) {
        vis[node] = 1;
        tin[node] = low[node] = timer;
        timer++;
        for (auto it : adj[node]) {
            if (it == parent) continue;
            if (vis[it] == 0) {
                dfs(it, node, vis, adj, tin, low, bridges);
                low[node] = min(low[it], low[node]);
                // node --- it
                if (low[it] > tin[node]) {
                    bridges.push_back({it, node});
                }
            }
            else {
                low[node] = min(low[node], low[it]);
            }
        }
    }
public:
    vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
        vector<int> adj[n];
        for (auto it : connections) {
            int u = it[0], v = it[1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        vector<int> vis(n, 0);
        int tin[n], low[n];
        vector<vector<int>> bridges;
        dfs(0, -1, vis, adj, tin, low, bridges);
        return bridges;
    }
};

Articulation Point : 
class Solution {
private:
    int timer = 1;
    void dfs(int node, int parent, vector<int> &vis,
             vector<int> adj[], int tin[], int low[], vector<vector<int>> &bridges) {
        vis[node] = 1;
        tin[node] = low[node] = timer;
        timer++;
        for (auto it : adj[node]) {
            if (it == parent) continue;
            if (vis[it] == 0) {
                dfs(it, node, vis, adj, tin, low, bridges);
                low[node] = min(low[it], low[node]);
                // node --- it
                if (low[it] > tin[node]) {
                    bridges.push_back({it, node});
                }
            }
            else {
                low[node] = min(low[node], low[it]);
            }
        }
    }
public:
    vector<vector<int>> criticalConnections(int n,
    vector<vector<int>>& connections) {
        vector<int> adj[n];
        for (auto it : connections) {
            int u = it[0], v = it[1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        vector<int> vis(n, 0);
        int tin[n], low[n];
        vector<vector<int>> bridges;
        dfs(0, -1, vis, adj, tin, low, bridges);
        return bridges;
    }
};

Trie : 
struct Node {
     Node* links[26];
    bool flag = false;

    bool containsKey(char ch) {
        return links[ch - 'a'] != NULL;
    }

    void put(char ch, Node* node) {
        links[ch - 'a'] = node;
    }

    Node* get(char ch) {
        return links[ch - 'a'];
    }
    void setEnd() {
        flag = true;
    }

    bool isEnd() {
        return flag;
    }
};
class Trie {
private:
    Node* root;

public:
       Trie() {
        root = new Node();
    }
    void insert(string word) {
        Node* node = root;
        for (int i = 0; i < word.length(); i++) {
            if (!node->containsKey(word[i])) {
                node->put(word[i], new Node());
            }
            node = node->get(word[i]);
        }
        node->setEnd();
    }

    bool search(string word) {
        Node* node = root;
        for (int i = 0; i < word.length(); i++) {
            if (!node->containsKey(word[i])) {
                return false;
            }
            node = node->get(word[i]);
        }
        return node->isEnd();
    }
    bool startsWith(string prefix) {
        Node* node = root;
        for (int i = 0; i < prefix.length(); i++) {
            if (!node->containsKey(prefix[i])) {
                return false;
            }
            node = node->get(prefix[i]);
        }
        return true;
    }
};

Treap : 
struct node {
    node *L, *R;
    int W, S, sm, V;    // W = weight/priority  ,   S = size of the treap   ,   sm = Sum of elements in the treap   ,   V = Value of the node
    bool F;             // Used to reverse the treap
    node(int x) {
        L = R = 0;      // Inotialise left and right pointers
        W = rand();
        S = 1;          
        sm = x;
        V = x;
        F = 0;          // Initially the array is straight 
    }
};

int size(node *treap) {
    return (treap == 0 ? 0 : treap->S);
}

int sum(node *treap) {
    return (treap == 0 ? 0 : treap->sm);
}

void push(node *treap) {                    // Used to reverse the treap, once the head of the treap is set to F = true.
    if (treap && treap->F) {
        treap->F = 0;
        swap(treap->L, treap->R);
        if (treap->L) treap->L->F ^= 1;
        if (treap->R) treap->R->F ^= 1;
    }
}

void split(node *treap, node *&left, node *&right, int k) {                 // Splits the trip into 2 parts - left and right
    if (treap == 0)
        left = right = 0;
    else {
        push(treap);
        if (size(treap->L) < k) {
            split(treap->R, treap->R, right, k - size(treap->L) - 1);
            left = treap;
        }
        else {
            split(treap->L, left, treap->L, k);
            right = treap;
        }
        treap->S = size(treap->L) + size(treap->R) + 1;
        treap->sm = sum(treap->L) + sum(treap->R) + treap->V;
    }
}

void merge(node *&treap, node *left, node *right) {                     // Merges the left and right parts and stores in the treap
    if (left == 0) treap = right;
    else if (right == 0) treap = left;
    else {
        push(left);
        push(right);
        if (left->W < right->W) {
            merge(left->R, left->R, right);
            treap = left;
        }
        else {
            merge(right->L, left, right->L);
            treap = right;
        }
        treap->S = size(treap->L) + size(treap->R) + 1;
        treap->sm = sum(treap->L) + sum(treap->R) + treap->V;
    }
}

void print(node *treap) {						// print treap
    if (treap == NULL) return;
    print(treap->left);
    cout << treap->value;
    print(treap->right);
}

int find_sum(node *treap, int k) {                                      // Computes the sum of the subarray
    if (treap == 0 || k == 0)
        return 0;
    else {
        push(treap);
        if (size(treap->L) < k) {
            return sum(treap->L) + treap->V + find_sum(treap->R, k - size(treap->L) - 1);
        }
        else {
            return find_sum(treap->L, k);
        }
    }
}

Monotonic Stack : 
   stack<pair<ll, ll>> st;
    vector<ll> nextGreaterA(n);
    st.push({1e9, n});
    for(ll i=n-1; i>=0; --i) {
        while(st.top().first <= a[i]) st.pop();
        nextGreaterA[i] = st.top().second;
        st.push({a[i], i});
    }

Cycle Detection :
bool detect_cycle_and_print(int node, vector<vector<int>>& adj, vector<int>& visited, vector<int>& inStack, vector<int>& parent, vector<int>& cycle) {
    visited[node] = 1;
    inStack[node] = 1;

    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) { 
            parent[neighbor] = node; 
            if (detect_cycle_and_print(neighbor, adj, visited, inStack, parent, cycle)) {
                return true;
            }
        } else if (inStack[neighbor]) { 
            cycle.pb(neighbor);
            int current = node;
            while (current != neighbor) {
                cycle.pb(current);
                current = parent[current];
            }
            cycle.pb(neighbor);
            reverse(cycle.begin(), cycle.end());
            return true;
        }
    }
    inStack[node] = 0; 
    return false;
}

Kosaraju :
class Solution
{
private:
    void dfs(int node, vector<int> &vis, vector<int> adj[],
             stack<int> &st) {
        vis[node] = 1;
        for (auto it : adj[node]) {
            if (!vis[it]) {
                dfs(it, vis, adj, st);
            }
        }
        st.push(node);
    }
private:
    void dfs3(int node, vector<int> &vis, vector<int> adjT[]) {
        vis[node] = 1;
        for (auto it : adjT[node]) {
            if (!vis[it]) {
                dfs3(it, vis, adjT);
            }
        }
    }
public:
    //Function to find number of strongly connected components in the graph.
    int kosaraju(int V, vector<int> adj[])
    {
        vector<int> vis(V, 0);
        stack<int> st;
        for (int i = 0; i < V; i++) {
            if (!vis[i]) {
                dfs(i, vis, adj, st);
            }
        }
        vector<int> adjT[V];
        for (int i = 0; i < V; i++) {
            vis[i] = 0;
            for (auto it : adj[i]) {
                // i -> it
                // it -> i
                adjT[it].push_back(i);
            }
        }
        int scc = 0;
        while (!st.empty()) {
            int node = st.top();
            st.pop();
            if (!vis[node]) {
                scc++;
                dfs3(node, vis, adjT);
            }
        }
        return scc;
    }
};
Kahn’s Algo:
vector<int> topoSort_kahans(vector<vector<int>>&adj , int n ){
    map<int,int> m;
    //  int n  = adj.size()-1;
    for(int i = 1;i<=n;i++) m[i] = 0;
    for(int i = 1;i<=n;i++){
        for(auto nbr : adj[i]){
            m[nbr]++;
        }
    }
    queue<int> q;
    for(auto it : m){
        if(it.second == 0){
            q.push(it.first);
        }
    }
    vector<int> ans;
    while(!q.empty()){
        auto frontNode = q.front();q.pop();
        ans.push_back(frontNode);

        for(auto nbr : adj[frontNode]){
            m[nbr]--;
            if(m[nbr] == 0){
                q.push(nbr);
            }
        }
    }

    return ans;
}
Kruskals -
class Solution {
public:
    int spanningTree(int V, vector<vector<int>> adj[]) {
        vector<pair<int, pair<int, int>>> edges;
        // Collect edges from adjacency list (only one direction to avoid duplicates)
        for (int i = 0; i < V; i++) {
            for (auto &it : adj[i]) {
                int u = i;
                int v = it[0];
                int w = it[1];
                    edges.push_back({w, {u, v}});
                
            }
        }
       
        DSU ds(V+2);
        sort(edges.begin(), edges.end()); // Sort edges by weight
        int ans = 0;
        // Iterate over the edges
        for (auto &edge : edges) {
            int u = edge.second.first;
            int v = edge.second.second;
            int w = edge.first;
            if (ds.findParent(u) != ds.findParent(v)) {
                ans += w; // Add edge weight to answer
                ds.unionBySize(u, v); // Union the sets
            }
        }

        return ans;
    }
};
// Prims + Print MST 
class Solution
{
    public:
    int spanningTree(int V, vector<vector<int>> adj[]) {
    vector<bool> vis(V, false);
    priority_queue<pair<int,pair<int,int>>, vector<pair<int, pair<int,int>>>, greater<pair<int, pair<int,int>>>> pq;
   vector<pair<int,int>> mst;
    pq.push({0, {0,-1}}); 
    int ans = 0; 

    while (!pq.empty()) {
        auto frontNode = pq.top();
        int weight = frontNode.first;
        int node = frontNode.second.first;
        int parent = frontNode.second.second;
        pq.pop();
        if (vis[node]) continue;
        vis[node] = true;
        mst.push_back({node,parent});
        ans += weight;
        for (auto &edge : adj[node]) {
            int adjNode = edge[0];
            int edgeWeight = edge[1];

            if (!vis[adjNode]) {
                pq.push({edgeWeight, {adjNode,node}});
            }
        }
    }

    return ans; 
}
};

Bitset Operations : 
bitset<8> decimalBitset(15);
bitset<8> stringBitset(string("1111"));
set() - Set the bit value at the given index to 1.
reset() - Set the bit value at a given index to 0.
flip() - Flip the bit value at the given index.
count() - Count the number of set bits.
test() - Returns the boolean value at the given index.
any() - Checks if any bit is set.
none() - Checks if none bit is set.
all() - Check if all bit is set.
size() - Returns the size of the bitset.
to_string() - Converts bitset to std::string.
to_ulong() - Converts bitset to unsigned long.
to_ullong() - Converts bitset to unsigned long long.
