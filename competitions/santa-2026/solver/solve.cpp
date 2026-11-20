// Santa 2026 - The Elf Calculator: Fast Multi-Strategy C++ Solver
// Combines:
//   0. Exact BFS for tiny/small puzzles
//   1. Multi-profile shared dictionary construction (small+med, mixed+large, divisor-adaptive, compact)
//   2. Backward A*/beam search using 1-op (div/sqrt), 2-op (q*m +/- d, q*e1), and 3-op (q*m +/- e1) steps
//   3. Inter-target dependency arborescence (T_j = p * T_i +/- delta or T_i / q) with DFS register reuse
//   4. Dead-code elimination (DCE) + full exact program verification
//
// Compile: g++ -O3 -march=native -fopenmp -std=c++20 solver/solve.cpp -o solver/solve

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using u64 = uint64_t;
using i64 = int64_t;
using u128 = unsigned __int128;

static constexpr int MAX_REGS = 48;
static constexpr int MAX_INSTS = 4096;
static constexpr size_t MAX_BYTES = 131072;

struct Inst {
    char op; // '+', '-', '*', '/'
    uint8_t i, j, d;
};

struct Puzzle {
    std::string id;
    std::vector<u64> starts;
    std::vector<u64> targets;
    u64 vlim = 0;
    int nregs = 48;
    std::string family;
    std::string tier;
};

// ---------------- CSV / JSON Parsing ----------------

static std::vector<u64> parse_u64_array(const std::string &s) {
    std::vector<u64> out;
    u64 cur = 0;
    bool in_num = false;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            cur = cur * 10 + (u64)(c - '0');
            in_num = true;
        } else if (in_num) {
            out.push_back(cur);
            cur = 0;
            in_num = false;
        }
    }
    if (in_num) out.push_back(cur);
    return out;
}

static std::vector<std::string> parse_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t idx = 0; idx < line.size(); ++idx) {
        char c = line[idx];
        if (in_quotes) {
            if (c == '"') {
                if (idx + 1 < line.size() && line[idx + 1] == '"') {
                    cur.push_back('"');
                    ++idx;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                fields.push_back(std::move(cur));
                cur.clear();
            } else if (c != '\r' && c != '\n') {
                cur.push_back(c);
            }
        }
    }
    fields.push_back(std::move(cur));
    return fields;
}

static std::vector<Puzzle> load_puzzles(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open " << path << "\n";
        std::exit(1);
    }
    std::string line;
    std::getline(in, line);
    auto header = parse_csv_line(line);
    std::unordered_map<std::string, int> col;
    for (int i = 0; i < (int)header.size(); ++i) col[header[i]] = i;

    std::vector<Puzzle> puzzles;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = parse_csv_line(line);
        if ((int)f.size() < 5) continue;
        Puzzle p;
        p.id = f[col["id"]];
        p.starts = parse_u64_array(f[col["starting_values"]]);
        p.targets = parse_u64_array(f[col["targets"]]);
        p.vlim = std::stoull(f[col["value_limit"]]);
        p.nregs = std::stoi(f[col["registers"]]);
        if (col.count("family") && col["family"] < (int)f.size()) p.family = f[col["family"]];
        if (col.count("design_tier") && col["design_tier"] < (int)f.size()) p.tier = f[col["design_tier"]];
        puzzles.push_back(std::move(p));
    }
    return puzzles;
}

static std::string program_to_json(const std::vector<Inst> &prog) {
    std::string s;
    s.reserve(prog.size() * 14 + 4);
    s.push_back('[');
    char buf[32];
    for (size_t k = 0; k < prog.size(); ++k) {
        if (k) s.push_back(',');
        int n = std::snprintf(buf, sizeof(buf), "[\"%c\",%u,%u,%u]",
                              prog[k].op, (unsigned)prog[k].i, (unsigned)prog[k].j, (unsigned)prog[k].d);
        s.append(buf, n);
    }
    s.push_back(']');
    return s;
}

// ---------------- Exact Verifier ----------------

static bool verify_program(const Puzzle &p, const std::vector<Inst> &prog, std::string *err = nullptr) {
    if ((int)prog.size() > MAX_INSTS) {
        if (err) *err = "too many instructions";
        return false;
    }
    std::array<u64, MAX_REGS> regs{};
    std::array<bool, MAX_REGS> def{};
    for (size_t k = 0; k < p.starts.size(); ++k) {
        regs[k] = p.starts[k];
        def[k] = true;
    }
    std::unordered_set<u64> rem(p.targets.begin(), p.targets.end());
    for (u64 v : p.starts) rem.erase(v);

    for (size_t step = 0; step < prog.size(); ++step) {
        const auto &in = prog[step];
        if (in.i >= p.nregs || in.j >= p.nregs || in.d >= p.nregs) {
            if (err) *err = "reg out of range";
            return false;
        }
        if (!def[in.i] || !def[in.j]) {
            if (err) *err = "undefined reg read at step " + std::to_string(step);
            return false;
        }
        u64 a = regs[in.i], b = regs[in.j];
        u128 res = 0;
        if (in.op == '+') {
            res = (u128)a + (u128)b;
        } else if (in.op == '-') {
            if (a <= b) {
                if (err) *err = "non-positive sub";
                return false;
            }
            res = (u128)(a - b);
        } else if (in.op == '*') {
            res = (u128)a * (u128)b;
        } else if (in.op == '/') {
            if (b == 0 || a % b != 0) {
                if (err) *err = "non-exact div";
                return false;
            }
            res = (u128)(a / b);
        } else {
            if (err) *err = "bad op";
            return false;
        }
        if (res < 1 || res > (u128)p.vlim) {
            if (err) *err = "out of value_limit";
            return false;
        }
        regs[in.d] = (u64)res;
        def[in.d] = true;
        rem.erase((u64)res);
    }
    if (!rem.empty()) {
        if (err) *err = "missed " + std::to_string(rem.size()) + " targets";
        return false;
    }
    if (program_to_json(prog).size() > MAX_BYTES) {
        if (err) *err = "exceeded byte limit";
        return false;
    }
    return true;
}

// ---------------- Dead-Code Elimination (DCE) ----------------
// Keeps exact register indices (valid because removing an unused write to d can only hurt
// if a later instruction reads d — which by backward liveness tracking it never does!).
// Wait: removing a write to d could leave an EARLIER value in d, which only matters if d is
// read AFTER this write (then this write WAS live!) — wait: what if d was overwritten later
// by a live instruction and read before that? Liveness tracks exact defining step per register!

static std::vector<Inst> dce_program(const Puzzle &p, const std::vector<Inst> &prog) {
    int n = (int)prog.size();
    if (n == 0) return prog;
    std::array<u64, MAX_REGS> regs{};
    std::array<int, MAX_REGS> def_step{};
    def_step.fill(-1);
    std::unordered_set<u64> target_set(p.targets.begin(), p.targets.end());
    std::unordered_set<u64> hit;
    for (size_t k = 0; k < p.starts.size(); ++k) {
        regs[k] = p.starts[k];
        def_step[k] = -2; // initial
        if (target_set.count(p.starts[k])) hit.insert(p.starts[k]);
    }
    std::vector<std::pair<int, int>> deps(n, {-1, -1});
    std::vector<bool> live(n, false);

    for (int s = 0; s < n; ++s) {
        const auto &in = prog[s];
        deps[s] = {def_step[in.i], def_step[in.j]};
        u64 a = regs[in.i], b = regs[in.j], res = 0;
        if (in.op == '+') res = a + b;
        else if (in.op == '-') res = a - b;
        else if (in.op == '*') res = a * b;
        else res = a / b;
        regs[in.d] = res;
        def_step[in.d] = s;
        if (target_set.count(res) && !hit.count(res)) {
            hit.insert(res);
            live[s] = true;
        }
    }
    // Backward propagate liveness
    for (int s = n - 1; s >= 0; --s) {
        if (!live[s]) continue;
        auto [d1, d2] = deps[s];
        if (d1 >= 0) live[d1] = true;
        if (d2 >= 0) live[d2] = true;
    }
    std::vector<Inst> out;
    out.reserve(n);
    for (int s = 0; s < n; ++s) {
        if (live[s]) out.push_back(prog[s]);
    }
    if (verify_program(p, out)) return out;
    return prog;
}

// ---------------- Helpers ----------------

static inline u64 isqrt_u64(u64 x) {
    u64 r = (u64)std::sqrt((long double)x);
    while ((u128)(r + 1) * (r + 1) <= x) ++r;
    while ((u128)r * r > x) --r;
    return r;
}

static inline double log2_u64(u64 x) {
    if (x <= 1) return 0.0;
    return std::log2((double)x);
}

// ---------------- Strategy 0: Tiny Exact / Fast BFS ----------------

static bool solve_tiny_bfs(const Puzzle &p, std::vector<Inst> &best_prog, int max_depth = 8) {
    std::unordered_set<u64> rem_init(p.targets.begin(), p.targets.end());
    for (u64 v : p.starts) rem_init.erase(v);
    if (rem_init.empty()) {
        best_prog.clear();
        return true;
    }
    if (p.vlim > 5000 && p.targets.size() > 2) return false;

    // Represent state as sorted vector of values (small depth -> regs simply append 0..47)
    struct Node {
        std::vector<u64> vals;
        std::vector<Inst> prog;
        uint32_t hit_mask;
    };
    std::vector<u64> need(rem_init.begin(), rem_init.end());
    if (need.size() > 20) return false;
    uint32_t full_mask = (1u << need.size()) - 1u;
    std::unordered_map<u64, int> t_idx;
    for (int i = 0; i < (int)need.size(); ++i) t_idx[need[i]] = i;

    std::vector<Node> cur;
    cur.push_back({p.starts, {}, 0});
    std::unordered_set<u64> seen_hashes;

    auto hash_state = [](const std::vector<u64> &v, uint32_t mask) -> u64 {
        std::vector<u64> s = v;
        std::sort(s.begin(), s.end());
        u64 h = 1469598103934665603ULL ^ mask;
        for (u64 x : s) {
            h ^= x;
            h *= 1099511628211ULL;
        }
        return h;
    };

    for (int depth = 1; depth <= max_depth; ++depth) {
        std::vector<Node> nxt;
        for (const auto &nd : cur) {
            int sz = (int)nd.vals.size();
            if (sz >= MAX_REGS) continue;
            uint8_t dst = (uint8_t)sz;
            int rem_need = __builtin_popcount(full_mask ^ nd.hit_mask);
            if (rem_need > max_depth - depth + 1) continue;

            std::unordered_set<u64> local_seen(nd.vals.begin(), nd.vals.end());
            auto try_val = [&](char op, uint8_t i, uint8_t j, u64 res) -> bool {
                if (res < 1 || res > p.vlim) return false;
                if (local_seen.count(res)) return false;
                local_seen.insert(res);
                uint32_t nmask = nd.hit_mask;
                auto it = t_idx.find(res);
                if (it != t_idx.end()) nmask |= (1u << it->second);
                if (nmask == full_mask) {
                    best_prog = nd.prog;
                    best_prog.push_back({op, i, j, dst});
                    return true;
                }
                if (depth < max_depth && nxt.size() < 40000) {
                    std::vector<u64> nv = nd.vals;
                    nv.push_back(res);
                    u64 h = hash_state(nv, nmask);
                    if (seen_hashes.insert(h).second) {
                        auto np = nd.prog;
                        np.push_back({op, i, j, dst});
                        nxt.push_back({std::move(nv), std::move(np), nmask});
                    }
                }
                return false;
            };

            for (int i = 0; i < sz; ++i) {
                for (int j = i; j < sz; ++j) {
                    u64 a = nd.vals[i], b = nd.vals[j];
                    if ((u128)a + b <= p.vlim && try_val('+', i, j, a + b)) return true;
                    if ((u128)a * b <= p.vlim && try_val('*', i, j, a * b)) return true;
                    if (a > b && try_val('-', i, j, a - b)) return true;
                    if (b > a && try_val('-', j, i, b - a)) return true;
                    if (a % b == 0 && try_val('/', i, j, a / b)) return true;
                    if (b % a == 0 && try_val('/', j, i, b / a)) return true;
                }
            }
        }
        cur = std::move(nxt);
        if (cur.empty()) break;
    }
    return false;
}

// ---------------- Shared Dictionary & 1-op Expression Tables ----------------

struct Expr1 {
    u64 val;
    char op;
    uint8_t a, b;
};

struct ModEntry {
    u64 rem;      // val % m
    int32_t idx;  // index in V (for mod_V) or E1 (for mod_E1)
    int8_t sign;  // +1 means T = q*m + d (so T%m == d%m); -1 means T = q*m - d (so T%m == (m - d%m)%m)
};

struct DictEnv {
    std::vector<u64> V;                 // values in registers 0..K-1
    std::vector<Inst> setup_prog;       // instructions building V
    std::unordered_map<u64, uint8_t> V_map; // val -> reg
    std::vector<Expr1> E1;              // distinct 1-op values not in V
    std::unordered_map<u64, int> E1_map; // val -> index in E1
    u64 vlim = 0;
    uint8_t reg_one = 255;
    uint8_t reg_two = 255;

    // For each m in V (index mi): sorted list of ModEntry for 2-op (d in V)
    std::vector<std::vector<ModEntry>> mod_V;
    // For selected large m in V: sorted list of ModEntry for 3-op (e1 in E1)
    std::vector<int> large_m_indices;
    std::vector<std::vector<ModEntry>> mod_E1;
    // Medium E1 values useful as exact 2-op divisors
    std::vector<int> div_E1_indices;

    // Scratch registers available: scratch1, scratch2 (and optionally parent regs in between)
    uint8_t scratch1 = 46;
    uint8_t scratch2 = 47;

    void build_tables(int max_e1_for_mod = 3400, int max_large_m = 36) {
        V_map.clear();
        V_map.reserve(V.size() * 2 + 8);
        reg_one = 255;
        reg_two = 255;
        for (int i = 0; i < (int)V.size(); ++i) {
            if (!V_map.count(V[i])) V_map[V[i]] = (uint8_t)i;
            if (V[i] == 1) reg_one = (uint8_t)i;
            if (V[i] == 2) reg_two = (uint8_t)i;
        }

        // Build E1 (all legal 1-op values from V)
        E1.clear();
        E1_map.clear();
        E1_map.reserve(4096);
        int K = (int)V.size();
        auto add_e1 = [&](u64 val, char op, uint8_t a, uint8_t b) {
            if (val < 1 || val > vlim) return;
            if (V_map.count(val) || E1_map.count(val)) return;
            int idx = (int)E1.size();
            E1_map[val] = idx;
            E1.push_back({val, op, a, b});
        };
        for (int i = 0; i < K; ++i) {
            for (int j = i; j < K; ++j) {
                u64 a = V[i], b = V[j];
                if ((u128)a + b <= vlim) add_e1(a + b, '+', (uint8_t)i, (uint8_t)j);
                if (a > b) add_e1(a - b, '-', (uint8_t)i, (uint8_t)j);
                else if (b > a) add_e1(b - a, '-', (uint8_t)j, (uint8_t)i);
                if ((u128)a * b <= vlim) add_e1(a * b, '*', (uint8_t)i, (uint8_t)j);
                if (b > 0 && a % b == 0) add_e1(a / b, '/', (uint8_t)i, (uint8_t)j);
                if (a > 0 && b % a == 0) add_e1(b / a, '/', (uint8_t)j, (uint8_t)i);
            }
        }

        // mod_V for all m in V with m >= 3
        mod_V.assign(K, {});
        for (int mi = 0; mi < K; ++mi) {
            u64 m = V[mi];
            if (m < 3) continue;
            auto &vec = mod_V[mi];
            vec.reserve(2 * K);
            for (int di = 0; di < K; ++di) {
                u64 d = V[di];
                u64 r = d % m;
                if (r != 0) {
                    vec.push_back({r, di, +1});
                    vec.push_back({m - r, di, -1});
                }
            }
            std::sort(vec.begin(), vec.end(), [](const ModEntry &x, const ModEntry &y) {
                if (x.rem != y.rem) return x.rem < y.rem;
                return x.idx < y.idx;
            });
            // Deduplicate identical (rem) keeping smaller d (preferring +1 for safety near vlim)
            std::vector<ModEntry> uniq;
            uniq.reserve(vec.size());
            for (const auto &e : vec) {
                if (uniq.empty() || uniq.back().rem != e.rem) {
                    uniq.push_back(e);
                } else {
                    // keep smaller V[idx] (or +1 over -1)
                    const auto &prev = uniq.back();
                    if (V[e.idx] < V[prev.idx] || (V[e.idx] == V[prev.idx] && e.sign > prev.sign)) {
                        uniq.back() = e;
                    }
                }
            }
            vec = std::move(uniq);
        }

        // Select top large m in V for 3-op mod_E1 table
        large_m_indices.clear();
        for (int mi = 0; mi < K; ++mi) {
            if (V[mi] >= 400) large_m_indices.push_back(mi);
        }
        std::sort(large_m_indices.begin(), large_m_indices.end(), [&](int a, int b) {
            return V[a] > V[b];
        });
        if ((int)large_m_indices.size() > max_large_m) {
            large_m_indices.resize(max_large_m);
        }

        // Pick smallest E1 entries for mod_E1 (smaller e1 -> safer against vlim and smaller residual)
        std::vector<int> e1_order(E1.size());
        std::iota(e1_order.begin(), e1_order.end(), 0);
        std::sort(e1_order.begin(), e1_order.end(), [&](int a, int b) {
            return E1[a].val < E1[b].val;
        });
        if ((int)e1_order.size() > max_e1_for_mod) e1_order.resize(max_e1_for_mod);

        mod_E1.assign(large_m_indices.size(), {});
        for (size_t li = 0; li < large_m_indices.size(); ++li) {
            u64 m = V[large_m_indices[li]];
            auto &vec = mod_E1[li];
            vec.reserve(e1_order.size() * 2);
            for (int ei : e1_order) {
                u64 d = E1[ei].val;
                u64 r = d % m;
                if (r != 0) {
                    vec.push_back({r, ei, +1});
                    vec.push_back({m - r, ei, -1});
                }
            }
            std::sort(vec.begin(), vec.end(), [](const ModEntry &x, const ModEntry &y) {
                if (x.rem != y.rem) return x.rem < y.rem;
                return x.idx < y.idx;
            });
            std::vector<ModEntry> uniq;
            uniq.reserve(vec.size());
            for (const auto &e : vec) {
                if (uniq.empty() || uniq.back().rem != e.rem) {
                    uniq.push_back(e);
                } else {
                    const auto &prev = uniq.back();
                    if (E1[e.idx].val < E1[prev.idx].val || (E1[e.idx].val == E1[prev.idx].val && e.sign > prev.sign)) {
                        uniq.back() = e;
                    }
                }
            }
            vec = std::move(uniq);
        }

        // E1 divisors (moderate size, fast exact division check)
        div_E1_indices.clear();
        for (int ei : e1_order) {
            if (E1[ei].val >= 11 && E1[ei].val <= 500000) {
                div_E1_indices.push_back(ei);
                if (div_E1_indices.size() >= 300) break;
            }
        }
    }
};

// ---------------- Dictionary Profiles ----------------

struct DictProfile {
    int K_target;          // max dictionary registers
    int small_count;       // how many small values (1, 2, 3, ...) to seed
    u64 med_lo, med_hi;    // medium multiplier band
    int med_count;
    u64 large_lo, large_hi; // large multiplier band
    bool use_target_divisors;
    uint32_t seed;
};

static std::vector<u64> extract_common_factors(const Puzzle &p) {
    // Count small-to-medium prime / composite divisors of targets
    std::unordered_map<u64, int> freq;
    for (u64 t : p.targets) {
        u64 x = t;
        for (u64 d = 2; d * d <= x && d <= 2000; ++d) {
            if (x % d == 0) {
                freq[d]++;
                u64 pw = d;
                while (x % d == 0) {
                    x /= d;
                    pw *= d;
                    if (pw <= 10000 && t % pw == 0) freq[pw]++;
                }
            }
        }
    }
    std::vector<std::pair<int, u64>> scored;
    for (auto &[d, c] : freq) {
        if (c >= 3) scored.push_back({c * (int)log2_u64(d), d});
    }
    std::sort(scored.begin(), scored.end(), std::greater<>());
    std::vector<u64> out;
    for (auto &[sc, d] : scored) {
        out.push_back(d);
        if (out.size() >= 12) break;
    }
    return out;
}

static DictEnv build_dictionary(const Puzzle &p, const DictProfile &prof) {
    DictEnv env;
    env.vlim = p.vlim;
    env.V = p.starts;

    std::unordered_set<u64> have(env.V.begin(), env.V.end());
    auto append_inst = [&](char op, uint8_t a, uint8_t b, u64 res) -> bool {
        if ((int)env.V.size() >= prof.K_target) return false;
        if (res < 1 || res > p.vlim || have.count(res)) return false;
        uint8_t dst = (uint8_t)env.V.size();
        env.setup_prog.push_back({op, a, b, dst});
        env.V.push_back(res);
        have.insert(res);
        return true;
    };

    // Always ensure 1 is present (1 op: reg0 / reg0)
    if (!have.count(1)) {
        append_inst('/', 0, 0, 1);
    }

    // Helper: try to build exact value `want` in 1 op from current V
    auto try_build_1op = [&](u64 want) -> bool {
        if (want < 1 || want > p.vlim || have.count(want)) return false;
        int K = (int)env.V.size();
        for (int i = 0; i < K; ++i) {
            for (int j = i; j < K; ++j) {
                u64 a = env.V[i], b = env.V[j];
                if ((u128)a + b == want) return append_inst('+', i, j, want);
                if (a > b && a - b == want) return append_inst('-', i, j, want);
                if (b > a && b - a == want) return append_inst('-', j, i, want);
                if ((u128)a * b == want) return append_inst('*', i, j, want);
                if (b > 0 && a % b == 0 && a / b == want) return append_inst('/', i, j, want);
                if (a > 0 && b % a == 0 && b / a == want) return append_inst('/', j, i, want);
            }
        }
        return false;
    };

    // 1. Seed small consecutive numbers 2, 3, 4, ... up to small_count
    //    Notice 2=1+1, 3=2+1, 4=2+2, 5=3+2, ... each takes 1 op!
    for (u64 v = 2; v <= (u64)prof.small_count && (int)env.V.size() < prof.K_target; ++v) {
        if (v > p.vlim) break;
        if (have.count(v)) continue;
        try_build_1op(v);
    }

    // 2. Optionally add common target divisors if buildable in 1 op (or stepping stone)
    if (prof.use_target_divisors) {
        auto facs = extract_common_factors(p);
        for (u64 f : facs) {
            if ((int)env.V.size() >= prof.K_target) break;
            try_build_1op(f);
        }
    }

    // 3. Stepping stones to reach med_lo and large_lo quickly
    //    Pick largest value < med_lo, square or multiply until we can reach the band!
    u64 max_t = p.targets.back();
    u64 eff_med_hi = std::min(prof.med_hi, std::min(p.vlim, max_t));
    u64 eff_large_hi = std::min(prof.large_hi, std::min(p.vlim, max_t));

    auto add_stepping_stones = [&](u64 target_band) {
        for (int iter = 0; iter < 5 && (int)env.V.size() < prof.K_target; ++iter) {
            u64 mx = *std::max_element(env.V.begin(), env.V.end());
            if (mx >= target_band) break;
            // Find best 1-op product <= min(vlim, target_band * 2) that moves toward sqrt(target_band) or target_band
            u64 best_v = 0;
            char best_op = '*';
            int bi = -1, bj = -1;
            u64 goal = std::min(target_band, p.vlim);
            int K = (int)env.V.size();
            for (int i = 0; i < K; ++i) {
                for (int j = i; j < K; ++j) {
                    u128 prod = (u128)env.V[i] * env.V[j];
                    if (prod > mx && prod <= (u128)p.vlim && prod <= (u128)goal * 2) {
                        u64 pv = (u64)prod;
                        // Prefer odd/coprime-ish near goal
                        if (best_v == 0 || std::llabs((i64)pv - (i64)goal) < std::llabs((i64)best_v - (i64)goal)) {
                            if (!have.count(pv)) {
                                best_v = pv; best_op = '*'; bi = i; bj = j;
                            }
                        }
                    }
                }
            }
            if (bi >= 0) append_inst(best_op, (uint8_t)bi, (uint8_t)bj, best_v);
            else break;
        }
    };

    if (prof.med_count > 0 && eff_med_hi > prof.med_lo) {
        add_stepping_stones(std::max<u64>(prof.med_lo / 10, 40));
    }

    // 4. Fill medium band [med_lo, eff_med_hi] preferring odd/coprime well-spaced values!
    std::mt19937_64 rng(prof.seed);
    auto fill_band = [&](u64 lo, u64 hi, int count) {
        if (count <= 0 || lo >= hi) return;
        for (int added = 0; added < count && (int)env.V.size() < prof.K_target; ++added) {
            int K = (int)env.V.size();
            u64 best_v = 0;
            double best_score = -1e18;
            char best_op = '+';
            int bi = -1, bj = -1;

            auto eval_cand = [&](u64 val, char op, int i, int j) {
                if (val < lo || val > hi || val > p.vlim || have.count(val)) return;
                // Score: prefer values coprime to existing band entries, odd, not div by 3/5, near upper part of band
                double sc = log2_u64(val) * 2.0;
                if (val % 2 == 1) sc += 2.5;
                if (val % 3 != 0) sc += 1.2;
                if (val % 5 != 0) sc += 0.8;
                if (val % 7 != 0) sc += 0.5;
                int shared_gcd = 0;
                for (u64 ev : env.V) {
                    if (ev >= lo / 2) {
                        u64 g = std::gcd(val, ev);
                        if (g > 1) shared_gcd += (g > 5 ? 3 : 1);
                    }
                }
                sc -= shared_gcd * 1.5;
                sc += (double)(rng() % 1000) * 1e-4;
                if (sc > best_score) {
                    best_score = sc; best_v = val; best_op = op; bi = i; bj = j;
                }
            };

            for (int i = 0; i < K; ++i) {
                for (int j = i; j < K; ++j) {
                    u64 a = env.V[i], b = env.V[j];
                    if ((u128)a + b <= p.vlim) eval_cand(a + b, '+', i, j);
                    if (a > b) eval_cand(a - b, '-', i, j);
                    else if (b > a) eval_cand(b - a, '-', j, i);
                    if ((u128)a * b <= p.vlim) eval_cand(a * b, '*', i, j);
                }
            }
            if (bi >= 0) {
                append_inst(best_op, (uint8_t)bi, (uint8_t)bj, best_v);
            } else {
                // Add a stepping stone to reach lo
                add_stepping_stones(lo);
                if ((int)env.V.size() == K) break;
            }
        }
    };

    fill_band(prof.med_lo, eff_med_hi, prof.med_count);

    // 5. Fill remaining slots in large band [large_lo, eff_large_hi]
    int rem_slots = prof.K_target - (int)env.V.size();
    if (rem_slots > 0 && eff_large_hi > prof.large_lo) {
        add_stepping_stones(prof.large_lo);
        rem_slots = prof.K_target - (int)env.V.size();
        fill_band(prof.large_lo, eff_large_hi, rem_slots);
    }
    // If still slots left, fill medium band more
    rem_slots = prof.K_target - (int)env.V.size();
    if (rem_slots > 0 && eff_med_hi > 10) {
        fill_band(std::min<u64>(prof.med_lo / 4, 20), eff_med_hi, rem_slots);
    }

    env.build_tables();
    return env;
}

// ---------------- Backward Beam Solver for Single Target X ----------------

// Represents one forward step from previous accumulator value `prev_val` -> `next_val`
enum StepKind : uint8_t {
    STEP_INIT_V,      // acc = V[a] (0 ops)
    STEP_INIT_E1,     // acc = V[a] op V[b] (1 op)
    STEP_INIT_E1_V,   // tmp = V[a] op1 V[b]; acc = tmp op2 V[c] (2 ops)
    STEP_INIT_V_E1,   // tmp = V[a] op1 V[b]; acc = V[c] op2 tmp (2 ops, for non-commutative - or /)
    STEP_INIT_E1_SQ,  // acc = V[a] op1 V[b]; acc = acc * acc (2 ops)
    STEP_OP_V,        // acc = acc op V[a] (1 op: *, +, -, /)
    STEP_SQ,          // acc = acc * acc (1 op)
    STEP_MUL_V_ADD_V, // acc = acc * V[m]; acc = acc +/- V[d] (2 ops)
    STEP_MUL_E1,      // tmp = V[a] op V[b]; acc = acc * tmp (2 ops)
    STEP_MUL_V_ADD_E1 // tmp = V[a] op V[b]; acc = acc * V[m]; acc = acc +/- tmp (3 ops)
};

struct Step {
    StepKind kind;
    char op1 = 0, op2 = 0;
    uint8_t a = 0, b = 0, c = 0;
    uint8_t cost = 0;
};

struct ChainResult {
    bool ok = false;
    int cost = 1000000;
    // Forward order: steps[0] initializes acc, steps[1..] transform acc
    std::vector<Step> steps;
};

static void emit_chain(const ChainResult &cr, uint8_t acc, uint8_t tmp, std::vector<Inst> &out) {
    assert(cr.ok && !cr.steps.empty());
    // Note: if steps[0] is STEP_INIT_V (0 ops), we don't write to `acc` yet; we remember `cur_reg = steps[0].a`
    // until the first real op writes to `acc`!
    uint8_t cur = acc;
    for (size_t idx = 0; idx < cr.steps.size(); ++idx) {
        const auto &st = cr.steps[idx];
        switch (st.kind) {
            case STEP_INIT_V:
                cur = st.a;
                break;
            case STEP_INIT_E1:
                out.push_back({st.op1, st.a, st.b, acc});
                cur = acc;
                break;
            case STEP_INIT_E1_V:
                out.push_back({st.op1, st.a, st.b, acc});
                out.push_back({st.op2, acc, st.c, acc});
                cur = acc;
                break;
            case STEP_INIT_V_E1:
                out.push_back({st.op1, st.a, st.b, tmp});
                out.push_back({st.op2, st.c, tmp, acc});
                cur = acc;
                break;
            case STEP_INIT_E1_SQ:
                out.push_back({st.op1, st.a, st.b, acc});
                out.push_back({'*', acc, acc, acc});
                cur = acc;
                break;
            case STEP_OP_V:
                out.push_back({st.op1, cur, st.a, acc});
                cur = acc;
                break;
            case STEP_SQ:
                out.push_back({'*', cur, cur, acc});
                cur = acc;
                break;
            case STEP_MUL_V_ADD_V:
                out.push_back({'*', cur, st.a, acc});
                out.push_back({st.op1, acc, st.b, acc});
                cur = acc;
                break;
            case STEP_MUL_E1:
                out.push_back({st.op1, st.a, st.b, tmp});
                out.push_back({'*', cur, tmp, acc});
                cur = acc;
                break;
            case STEP_MUL_V_ADD_E1:
                out.push_back({st.op1, st.a, st.b, tmp});
                out.push_back({'*', cur, st.c, acc});
                out.push_back({st.op2, acc, tmp, acc});
                cur = acc;
                break;
        }
    }
    // If cur != acc (meaning cost == 0, value was already in V[cur]), no instruction needed unless caller wants it in acc.
    (void)cur;
}

// Check if `val` can be synthesized in 0, 1, or 2 ops directly from DictEnv
static bool check_base_case(const DictEnv &env, u64 val, Step &out_step) {
    // Cost 0: in V
    auto itv = env.V_map.find(val);
    if (itv != env.V_map.end()) {
        out_step = {STEP_INIT_V, 0, 0, itv->second, 0, 0, 0};
        return true;
    }
    // Cost 1: in E1
    auto ite = env.E1_map.find(val);
    if (ite != env.E1_map.end()) {
        const auto &e = env.E1[ite->second];
        out_step = {STEP_INIT_E1, e.op, 0, e.a, e.b, 0, 1};
        return true;
    }
    // Cost 2: exact square of E1
    u64 sq = isqrt_u64(val);
    if ((u128)sq * sq == val) {
        auto its = env.E1_map.find(sq);
        if (its != env.E1_map.end()) {
            const auto &e = env.E1[its->second];
            out_step = {STEP_INIT_E1_SQ, e.op, '*', e.a, e.b, 0, 2};
            return true;
        }
    }
    // Cost 2: E1 op V or V op E1
    int K = (int)env.V.size();
    for (int i = 0; i < K; ++i) {
        u64 v = env.V[i];
        // e + v = val => e = val - v
        if (val > v) {
            auto it = env.E1_map.find(val - v);
            if (it != env.E1_map.end()) {
                const auto &e = env.E1[it->second];
                out_step = {STEP_INIT_E1_V, e.op, '+', e.a, e.b, (uint8_t)i, 2};
                return true;
            }
        }
        // e - v = val => e = val + v
        if ((u128)val + v <= env.vlim) {
            auto it = env.E1_map.find(val + v);
            if (it != env.E1_map.end()) {
                const auto &e = env.E1[it->second];
                out_step = {STEP_INIT_E1_V, e.op, '-', e.a, e.b, (uint8_t)i, 2};
                return true;
            }
        }
        // v - e = val => e = v - val
        if (v > val) {
            auto it = env.E1_map.find(v - val);
            if (it != env.E1_map.end()) {
                const auto &e = env.E1[it->second];
                out_step = {STEP_INIT_V_E1, e.op, '-', e.a, e.b, (uint8_t)i, 2};
                return true;
            }
        }
        // e * v = val => e = val / v
        if (v >= 2 && val % v == 0) {
            auto it = env.E1_map.find(val / v);
            if (it != env.E1_map.end()) {
                const auto &e = env.E1[it->second];
                out_step = {STEP_INIT_E1_V, e.op, '*', e.a, e.b, (uint8_t)i, 2};
                return true;
            }
        }
        // e / v = val => e = val * v
        if (v >= 2 && (u128)val * v <= env.vlim) {
            auto it = env.E1_map.find(val * v);
            if (it != env.E1_map.end()) {
                const auto &e = env.E1[it->second];
                out_step = {STEP_INIT_E1_V, e.op, '/', e.a, e.b, (uint8_t)i, 2};
                return true;
            }
        }
    }
    return false;
}

static ChainResult solve_single_value(
    const DictEnv &env,
    u64 target,
    int max_expansions = 180,
    int cost_limit = 64,
    double bits_per_op = 6.2
) {
    ChainResult best;
    Step base;
    if (check_base_case(env, target, base)) {
        best.ok = true;
        best.cost = base.cost;
        best.steps = {base};
        return best;
    }

    struct BNode {
        u64 val;
        int cost;
        int parent; // index in nodes
        Step fwd_step; // forward step from this node's val to parent's val
    };
    std::vector<BNode> nodes;
    nodes.reserve(max_expansions * 8 + 32);
    nodes.push_back({target, 0, -1, {}});

    struct PQItem {
        double f;
        int cost;
        int id;
        bool operator>(const PQItem &o) const {
            if (f != o.f) return f > o.f;
            return cost > o.cost;
        }
    };
    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
    pq.push({log2_u64(target) / bits_per_op, 0, 0});

    std::unordered_map<u64, int> best_g;
    best_g.reserve(max_expansions * 16 + 64);
    best_g[target] = 0;

    int best_total = cost_limit;
    int best_leaf_node = -1;
    Step best_leaf_init{};

    auto push_state = [&](u64 nval, int ncost, int par_id, const Step &st) {
        if (nval < 1 || ncost >= best_total) return;
        // Immediate base check
        Step init_st;
        if (check_base_case(env, nval, init_st)) {
            int tot = ncost + init_st.cost;
            if (tot < best_total) {
                best_total = tot;
                int nid = (int)nodes.size();
                nodes.push_back({nval, ncost, par_id, st});
                best_leaf_node = nid;
                best_leaf_init = init_st;
            }
            return;
        }
        auto it = best_g.find(nval);
        if (it != best_g.end() && it->second <= ncost) return;
        best_g[nval] = ncost;
        double est = ncost + log2_u64(nval) / bits_per_op;
        if (est >= (double)best_total + 1.5) return;
        int nid = (int)nodes.size();
        nodes.push_back({nval, ncost, par_id, st});
        pq.push({est, ncost, nid});
    };

    int K = (int)env.V.size();
    int expanded = 0;
    while (!pq.empty() && expanded < max_expansions) {
        auto top = pq.top();
        pq.pop();
        const u64 val = nodes[top.id].val;
        const int cost = nodes[top.id].cost;
        if (cost >= best_total) continue;
        auto itg = best_g.find(val);
        if (itg != best_g.end() && itg->second < cost) continue;
        ++expanded;

        // 1-op transitions: exact division by m in V
        for (int mi = 0; mi < K; ++mi) {
            u64 m = env.V[mi];
            if (m >= 2 && val % m == 0) {
                Step st{STEP_OP_V, '*', 0, (uint8_t)mi, 0, 0, 1};
                push_state(val / m, cost + 1, top.id, st);
            }
        }
        // 1-op transition: exact square root
        if (val >= 4) {
            u64 sq = isqrt_u64(val);
            if ((u128)sq * sq == val) {
                Step st{STEP_SQ, '*', 0, 0, 0, 0, 1};
                push_state(sq, cost + 1, top.id, st);
            }
        }

        // 2-op transitions: val = q * m +/- d (m, d in V)
        if (cost + 2 < best_total) {
            for (int mi = 0; mi < K; ++mi) {
                u64 m = env.V[mi];
                if (m < 3) continue;
                u64 r = val % m;
                if (r == 0) continue;
                const auto &mv = env.mod_V[mi];
                auto it = std::lower_bound(mv.begin(), mv.end(), r,
                    [](const ModEntry &e, u64 key) { return e.rem < key; });
                if (it != mv.end() && it->rem == r) {
                    u64 d = env.V[it->idx];
                    if (it->sign > 0) {
                        if (val > d) {
                            u64 q = (val - d) / m;
                            if (q >= 1 && q < val) {
                                Step st{STEP_MUL_V_ADD_V, '+', 0, (uint8_t)mi, (uint8_t)it->idx, 0, 2};
                                push_state(q, cost + 2, top.id, st);
                            }
                        }
                    } else {
                        if ((u128)val + d <= env.vlim) {
                            u64 q = (val + d) / m;
                            if (q >= 1 && q < val) {
                                Step st{STEP_MUL_V_ADD_V, '-', 0, (uint8_t)mi, (uint8_t)it->idx, 0, 2};
                                push_state(q, cost + 2, top.id, st);
                            }
                        }
                    }
                }
            }
            // 2-op exact div by E1
            for (int ei : env.div_E1_indices) {
                u64 d = env.E1[ei].val;
                if (val > d && val % d == 0) {
                    const auto &e = env.E1[ei];
                    Step st{STEP_MUL_E1, e.op, 0, e.a, e.b, 0, 2};
                    push_state(val / d, cost + 2, top.id, st);
                }
            }
        }

        // 3-op transitions: val = q * m +/- e1 (large m in V, e1 in E1)
        if (cost + 3 < best_total && val >= 2000) {
            for (size_t li = 0; li < env.large_m_indices.size(); ++li) {
                int mi = env.large_m_indices[li];
                u64 m = env.V[mi];
                if (m >= val) continue;
                u64 r = val % m;
                if (r == 0) continue;
                const auto &me = env.mod_E1[li];
                auto it = std::lower_bound(me.begin(), me.end(), r,
                    [](const ModEntry &e, u64 key) { return e.rem < key; });
                if (it != me.end() && it->rem == r) {
                    const auto &e = env.E1[it->idx];
                    if (it->sign > 0) {
                        if (val > e.val) {
                            u64 q = (val - e.val) / m;
                            if (q >= 1 && q < val) {
                                Step st{STEP_MUL_V_ADD_E1, e.op, '+', e.a, e.b, (uint8_t)mi, 3};
                                push_state(q, cost + 3, top.id, st);
                            }
                        }
                    } else {
                        if ((u128)val + e.val <= env.vlim) {
                            u64 q = (val + e.val) / m;
                            if (q >= 1 && q < val) {
                                Step st{STEP_MUL_V_ADD_E1, e.op, '-', e.a, e.b, (uint8_t)mi, 3};
                                push_state(q, cost + 3, top.id, st);
                            }
                        }
                    }
                }
            }
        }

        // Guaranteed binary fallback (if 1 and 2 are in V) so beam always finishes
        if (env.reg_two != 255 && env.reg_one != 255 && val >= 3 && (val & 1)) {
            Step stp{STEP_MUL_V_ADD_V, '+', 0, env.reg_two, env.reg_one, 0, 2};
            push_state(val >> 1, cost + 2, top.id, stp);
            if (val + 1 <= env.vlim) {
                Step stm{STEP_MUL_V_ADD_V, '-', 0, env.reg_two, env.reg_one, 0, 2};
                push_state((val + 1) >> 1, cost + 2, top.id, stm);
            }
        }
    }

    if (best_leaf_node >= 0) {
        best.ok = true;
        best.cost = best_total;
        best.steps.push_back(best_leaf_init);
        int cur = best_leaf_node;
        while (nodes[cur].parent >= 0) {
            best.steps.push_back(nodes[cur].fwd_step);
            cur = nodes[cur].parent;
        }
        return best;
    }

    // Hard fallback: binary double-and-add (guaranteed valid if 1 and 2 in V)
    if (env.reg_one != 255 && env.reg_two != 255) {
        best.ok = true;
        best.steps.clear();
        best.steps.push_back({STEP_INIT_V, 0, 0, env.reg_one, 0, 0, 0});
        int top_bit = 63 - __builtin_clzll(target);
        int c = 0;
        for (int b = top_bit - 1; b >= 0; --b) {
            if ((target >> b) & 1ULL) {
                best.steps.push_back({STEP_MUL_V_ADD_V, '+', 0, env.reg_two, env.reg_one, 0, 2});
                c += 2;
            } else {
                best.steps.push_back({STEP_OP_V, '*', 0, env.reg_two, 0, 0, 1});
                c += 1;
            }
        }
        best.cost = c;
    }
    return best;
}

// ---------------- Inter-Target Relational Edge & Tree Scheduling ----------------

enum EdgeType : uint8_t {
    EDGE_NONE = 0,
    EDGE_ADD_DELTA,      // Tj = Ti + delta
    EDGE_SUB_DELTA,      // Tj = Ti - delta
    EDGE_MUL_P_ADD_D,    // Tj = Ti * p + delta (delta may be 0)
    EDGE_MUL_P_SUB_D,    // Tj = Ti * p - delta (requires Ti*p <= vlim)
    EDGE_DIV_Q,          // Tj = Ti / q
    EDGE_RAT_ADD_D,      // Tj = (Ti * p + delta) / q (q in V)
    EDGE_RAT_SUB_D       // Tj = (Ti * p - delta) / q (q in V)
};

struct RelEdge {
    EdgeType type = EDGE_NONE;
    int cost = 1000000;
    u64 p = 1;
    u64 delta = 0;
    uint8_t q_reg = 255;
    ChainResult p_chain;
    ChainResult d_chain;
};

// Check cheap ways to derive target `tj` from an already-computed target `ti` in register `parent_reg`
static RelEdge best_rel_edge(const DictEnv &env, u64 ti, u64 tj, int max_cost) {
    RelEdge best;
    best.cost = max_cost;
    if (ti == tj) return best;

    // 1. Direct difference: tj = ti + delta or ti - delta
    u64 diff = (tj > ti) ? (tj - ti) : (ti - tj);
    if (diff <= 1000000000ULL) {
        auto cr = solve_single_value(env, diff, 45, best.cost - 1);
        if (cr.ok && cr.cost + 1 < best.cost) {
            best.type = (tj > ti) ? EDGE_ADD_DELTA : EDGE_SUB_DELTA;
            best.cost = cr.cost + 1;
            best.delta = diff;
            best.d_chain = std::move(cr);
        }
    }

    // 2. Exact division: tj = ti / q
    if (ti > tj && ti % tj == 0) {
        u64 q = ti / tj;
        auto cr = solve_single_value(env, q, 45, best.cost - 1);
        if (cr.ok && cr.cost + 1 < best.cost) {
            best.type = EDGE_DIV_Q;
            best.cost = cr.cost + 1;
            best.p = q;
            best.p_chain = std::move(cr);
        }
    }

    // 3. Scaled + small delta: tj = ti * p +/- delta (for p >= 2)
    if (tj > ti) {
        u64 q0 = tj / ti;
        for (u64 p_cand : {q0, q0 + 1}) {
            if (p_cand < 2 || p_cand > 500000) continue;
            u128 prod = (u128)ti * p_cand;
            if (prod > env.vlim) continue;
            u64 pv = (u64)prod;
            if (pv == tj) {
                auto pcr = solve_single_value(env, p_cand, 45, best.cost - 1);
                if (pcr.ok && pcr.cost + 1 < best.cost) {
                    best.type = EDGE_MUL_P_ADD_D;
                    best.cost = pcr.cost + 1;
                    best.p = p_cand;
                    best.delta = 0;
                    best.p_chain = std::move(pcr);
                }
            } else {
                u64 d = (tj > pv) ? (tj - pv) : (pv - tj);
                if (d < ti / 2 && d <= 50000000ULL) {
                    auto pcr = solve_single_value(env, p_cand, 35, best.cost - 2);
                    if (pcr.ok) {
                        int rem_budget = best.cost - 2 - pcr.cost;
                        if (rem_budget >= 0) {
                            auto dcr = solve_single_value(env, d, 35, rem_budget + 1);
                            if (dcr.ok && pcr.cost + dcr.cost + 2 < best.cost) {
                                best.type = (tj > pv) ? EDGE_MUL_P_ADD_D : EDGE_MUL_P_SUB_D;
                                best.cost = pcr.cost + dcr.cost + 2;
                                best.p = p_cand;
                                best.delta = d;
                                best.p_chain = std::move(pcr);
                                best.d_chain = std::move(dcr);
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Rational relation: q * tj = p * ti +/- delta for small q in V (2..8)
    for (u64 q_val : {2ULL, 3ULL, 4ULL, 5ULL, 6ULL, 8ULL}) {
        auto itq = env.V_map.find(q_val);
        if (itq == env.V_map.end()) continue;
        u128 qtj = (u128)q_val * tj;
        if (qtj > env.vlim) continue;
        u64 p0 = (u64)(qtj / ti);
        for (u64 p_cand : {p0, p0 + 1}) {
            if (p_cand < 1 || p_cand > 200 || std::gcd(p_cand, q_val) > 1) continue;
            u128 pti = (u128)p_cand * ti;
            if (pti > env.vlim) continue;
            u64 d = (qtj > pti) ? (u64)(qtj - pti) : (u64)(pti - qtj);
            if (d > 500000ULL) continue;
            if (d == 0) {
                auto pcr = solve_single_value(env, p_cand, 30, best.cost - 2);
                if (pcr.ok && pcr.cost + 2 < best.cost) {
                    best.type = EDGE_RAT_ADD_D;
                    best.cost = pcr.cost + 2;
                    best.p = p_cand;
                    best.delta = 0;
                    best.q_reg = itq->second;
                    best.p_chain = std::move(pcr);
                }
            } else {
                int extra_ops = (p_cand == 1) ? 2 : 3;
                if (extra_ops >= best.cost) continue;
                ChainResult pcr;
                if (p_cand == 1) { pcr.ok = true; pcr.cost = 0; }
                else pcr = solve_single_value(env, p_cand, 25, best.cost - extra_ops);
                if (pcr.ok && pcr.cost + extra_ops < best.cost) {
                    auto dcr = solve_single_value(env, d, 25, best.cost - extra_ops - pcr.cost);
                    if (dcr.ok && pcr.cost + dcr.cost + extra_ops < best.cost) {
                        best.type = (qtj > pti) ? EDGE_RAT_ADD_D : EDGE_RAT_SUB_D;
                        best.cost = pcr.cost + dcr.cost + extra_ops;
                        best.p = p_cand;
                        best.delta = d;
                        best.q_reg = itq->second;
                        best.p_chain = std::move(pcr);
                        best.d_chain = std::move(dcr);
                    }
                }
            }
        }
    }
    return best;
}

static void emit_rel_edge(
    const RelEdge &e,
    uint8_t parent_reg,
    uint8_t dst_reg,
    uint8_t scratch1,
    uint8_t scratch2,
    std::vector<Inst> &out
) {
    auto resolve_chain = [&](const ChainResult &cr, uint8_t target_reg, uint8_t tmp_reg) -> uint8_t {
        if (cr.cost == 0 && cr.steps.size() == 1 && cr.steps[0].kind == STEP_INIT_V) {
            return cr.steps[0].a;
        }
        emit_chain(cr, target_reg, tmp_reg, out);
        return target_reg;
    };

    if (e.type == EDGE_ADD_DELTA || e.type == EDGE_SUB_DELTA) {
        uint8_t dr = resolve_chain(e.d_chain, scratch1, scratch2);
        out.push_back({e.type == EDGE_ADD_DELTA ? '+' : '-', parent_reg, dr, dst_reg});
    } else if (e.type == EDGE_DIV_Q) {
        uint8_t qr = resolve_chain(e.p_chain, scratch1, scratch2);
        out.push_back({'/', parent_reg, qr, dst_reg});
    } else if (e.type == EDGE_MUL_P_ADD_D || e.type == EDGE_MUL_P_SUB_D) {
        uint8_t pr = resolve_chain(e.p_chain, scratch1, scratch2);
        if (e.delta == 0) {
            out.push_back({'*', parent_reg, pr, dst_reg});
        } else {
            out.push_back({'*', parent_reg, pr, dst_reg});
            uint8_t dr = resolve_chain(e.d_chain, scratch1, scratch2);
            out.push_back({e.type == EDGE_MUL_P_ADD_D ? '+' : '-', dst_reg, dr, dst_reg});
        }
    } else if (e.type == EDGE_RAT_ADD_D || e.type == EDGE_RAT_SUB_D) {
        uint8_t cur_reg = parent_reg;
        if (e.p > 1) {
            uint8_t pr = resolve_chain(e.p_chain, scratch1, scratch2);
            out.push_back({'*', parent_reg, pr, dst_reg});
            cur_reg = dst_reg;
        }
        if (e.delta > 0) {
            uint8_t dr = resolve_chain(e.d_chain, scratch1, scratch2);
            out.push_back({e.type == EDGE_RAT_ADD_D ? '+' : '-', cur_reg, dr, dst_reg});
            cur_reg = dst_reg;
        }
        out.push_back({'/', cur_reg, e.q_reg, dst_reg});
    }
}

// Solve a puzzle with a given DictProfile (max_live_parents=0 gives K up to 46 for max dictionary power!)
static bool solve_with_profile(
    const Puzzle &p,
    const DictProfile &prof,
    std::vector<Inst> &out_prog,
    int max_exp = 240,
    int max_live_parents = 4
) {
    int reserved = (max_live_parents > 0) ? (3 + max_live_parents) : 2;
    int max_dict_k = std::min(prof.K_target, MAX_REGS - reserved);
    max_dict_k = std::max(max_dict_k, std::min(MAX_REGS - reserved, (int)p.starts.size() + 3));
    DictProfile adj = prof;
    adj.K_target = max_dict_k;

    DictEnv env = build_dictionary(p, adj);
    uint8_t scratch0 = (max_live_parents > 0) ? 45 : 46;
    uint8_t scratch1 = 46;
    uint8_t scratch2 = 47;

    std::unordered_set<u64> covered(env.V.begin(), env.V.end());
    std::vector<u64> rem_targets;
    for (u64 t : p.targets) {
        if (!covered.count(t)) rem_targets.push_back(t);
    }
    if (rem_targets.empty()) {
        out_prog = dce_program(p, env.setup_prog);
        return verify_program(p, out_prog);
    }

    int N = (int)rem_targets.size();
    std::vector<ChainResult> root_chains(N);
    for (int i = 0; i < N; ++i) {
        root_chains[i] = solve_single_value(env, rem_targets[i], max_exp, 80);
        if (!root_chains[i].ok) return false;
    }

    if (max_live_parents == 0) {
        std::vector<Inst> prog = env.setup_prog;
        for (int i = 0; i < N; ++i) {
            emit_chain(root_chains[i], scratch1, scratch2, prog);
        }
        prog = dce_program(p, prog);
        if (!verify_program(p, prog)) return false;
        out_prog = std::move(prog);
        return true;
    }

    // Build relational edges among targets
    std::vector<std::vector<std::pair<int, RelEdge>>> best_in(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            u64 ti = rem_targets[i], tj = rem_targets[j];
            bool plausible = false;
            u64 diff = (tj > ti) ? (tj - ti) : (ti - tj);
            if (diff <= 50000000ULL) plausible = true;
            else if (ti > tj && ti % tj == 0) plausible = true;
            else if (tj > ti) {
                u64 q = tj / ti;
                if (q >= 2 && q <= 500000) {
                    u64 r1 = tj - q * ti;
                    u64 r2 = (q + 1) * ti - tj;
                    if (std::min(r1, r2) <= 50000000ULL) plausible = true;
                }
            }
            if (!plausible) {
                // Quick check for small rational q*tj ~= p*ti
                for (u64 qv = 2; qv <= 8; ++qv) {
                    u128 qtj = (u128)qv * tj;
                    if (qtj > env.vlim) break;
                    u64 p0 = (u64)(qtj / ti);
                    if (p0 >= 1 && p0 <= 200) {
                        u128 r1 = qtj - (u128)p0 * ti;
                        u128 r2 = (u128)(p0 + 1) * ti - qtj;
                        if (std::min(r1, r2) <= 500000ULL) { plausible = true; break; }
                    }
                }
            }
            if (!plausible) continue;
            RelEdge re = best_rel_edge(env, ti, tj, root_chains[j].cost);
            if (re.type != EDGE_NONE && re.cost < root_chains[j].cost) {
                best_in[j].push_back({i, std::move(re)});
            }
        }
    }

    // Greedy arborescence construction (depth-capped at max_live_parents - 1 to guarantee register availability!)
    std::vector<bool> used(N, false);
    std::vector<int> parent(N, -1); // -1 means built from dict root
    std::vector<int> depth(N, 0);
    std::vector<RelEdge> chosen_edge(N);
    std::vector<std::vector<int>> children(N);
    std::vector<int> roots;

    for (int step = 0; step < N; ++step) {
        int best_j = -1, best_par = -1;
        int best_c = 1000000;
        const RelEdge *best_re_ptr = nullptr;
        for (int j = 0; j < N; ++j) {
            if (used[j]) continue;
            // Option 1: from dict root
            if (root_chains[j].cost < best_c) {
                best_c = root_chains[j].cost;
                best_j = j;
                best_par = -1;
                best_re_ptr = nullptr;
            }
            // Option 2: from an already-used node i (keeping depth < max_live_parents)
            for (const auto &[i, re] : best_in[j]) {
                if (!used[i]) continue;
                if (depth[i] + 1 >= max_live_parents) continue;
                if (re.cost < best_c) {
                    best_c = re.cost;
                    best_j = j;
                    best_par = i;
                    best_re_ptr = &re;
                }
            }
        }
        used[best_j] = true;
        parent[best_j] = best_par;
        if (best_par == -1) {
            depth[best_j] = 0;
            roots.push_back(best_j);
        } else {
            depth[best_j] = depth[best_par] + 1;
            chosen_edge[best_j] = *best_re_ptr;
            children[best_par].push_back(best_j);
        }
    }

    // Emit program using DFS over the forest!
    // Pool of parent registers: [env.V.size() .. 44] (at least 4 registers!)
    std::vector<Inst> prog = env.setup_prog;
    int base_pool = (int)env.V.size();
    // Leaf nodes write into scratch0 (disjoint from scratch1 and scratch2!)
    std::function<void(int, int)> dfs_emit = [&](int u, int d_slot) {
        bool has_kids = !children[u].empty();
        uint8_t my_reg = has_kids ? (uint8_t)(base_pool + d_slot) : scratch0;
        if (parent[u] == -1) {
            emit_chain(root_chains[u], my_reg, scratch2, prog);
        } else {
            uint8_t par_reg = (uint8_t)(base_pool + d_slot - 1);
            emit_rel_edge(chosen_edge[u], par_reg, my_reg, scratch1, scratch2, prog);
        }
        // Visit leaf children FIRST (writing into scratch0), then internal children
        for (int c : children[u]) {
            if (children[c].empty()) {
                emit_rel_edge(chosen_edge[c], my_reg, scratch0, scratch1, scratch2, prog);
            }
        }
        for (int c : children[u]) {
            if (!children[c].empty()) {
                dfs_emit(c, d_slot + 1);
            }
        }
    };

    for (int r : roots) {
        dfs_emit(r, 0);
    }

    prog = dce_program(p, prog);
    if (!verify_program(p, prog)) return false;
    out_prog = std::move(prog);
    return true;
}

// ---------------- Portfolio Solver per Puzzle ----------------

static std::vector<Inst> solve_puzzle(const Puzzle &p) {
    std::vector<Inst> best;

    // Strategy 0: Exact BFS for small/tiny puzzles
    if (p.vlim <= 5000 || (p.targets.size() <= 4 && p.targets.back() <= 500)) {
        std::vector<Inst> tiny;
        if (solve_tiny_bfs(p, tiny, 8) && verify_program(p, tiny)) {
            best = std::move(tiny);
            if (best.size() <= p.targets.size() + 2) return best;
        }
    }

    u64 max_t = p.targets.back();
    int n_t = (int)p.targets.size();

    // Portfolio of (DictProfile, max_live_parents) covering all 5 puzzle families!
    std::vector<std::pair<DictProfile, int>> profiles;
    // --- Full K=46 dictionary (zero parent reserve) for independent / high-bit puzzles ---
    // 1. K=46 Optimal 3-op sweet spot (large in 45k..200k + medium 1200..5500)
    profiles.push_back({{46, 10, 1200, 5500, 10, 45000, 200000, false, 11}, 0});
    // 2. K=46 Heavy-Large (30+ values in 30k..150k -> ~2.5 3-op hits/step!)
    profiles.push_back({{46, 9, 800, 3500, 5, 30000, 150000, false, 22}, 0});
    // 3. K=46 Ultra-Large (values in 80k..300k for 55-60 bit targets)
    profiles.push_back({{46, 9, 1000, 4000, 5, 80000, 300000, false, 33}, 0});
    // 4. K=46 Medium-Dense (values in 1200..5000 -> ~2.0 2-op hits/step!)
    profiles.push_back({{46, 12, 1200, 5000, 34, 0, 0, false, 44}, 0});

    // --- K=41 + 4 parent target slots (for shared / multiplicative / mixed) ---
    // 5. Balanced Med+Large + Target Tree
    profiles.push_back({{41, 10, 1000, 4500, 10, 35000, 170000, false, 101}, 4});
    // 6. Divisor-Adaptive + Target Tree
    profiles.push_back({{41, 12, 800, 4000, 12, 30000, 150000, true, 202}, 4});
    // 7. Medium-Dense + Divisor-Adaptive + Target Tree
    profiles.push_back({{41, 12, 1000, 4500, 28, 0, 0, true, 303}, 4});
    // 8. Compact Dict + Target Tree
    profiles.push_back({{28, 10, 400, 3000, 12, 20000, 100000, true, 404}, 4});
    // 9. Smaller Dicts for lower target count / smaller magnitude puzzles
    if (n_t <= 22 || max_t < 10000000000ULL) {
        profiles.push_back({{20, 9, 150, 1500, 11, 0, 0, true, 505}, 4});
        profiles.push_back({{13, 8, 30, 400, 5, 0, 0, true, 606}, 4});
    }

    for (const auto &[prof, mlp] : profiles) {
        std::vector<Inst> cand;
        if (solve_with_profile(p, prof, cand, 240, mlp)) {
            if (best.empty() || cand.size() < best.size()) {
                best = std::move(cand);
            }
        }
    }

    // Guaranteed ultra-safe binary fallback if somehow nothing succeeded
    if (best.empty()) {
        DictProfile fb{12, 4, 0, 0, 0, 0, 0, false, 999};
        solve_with_profile(p, fb, best, 30, 0);
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <puzzles.csv> <out_submission.csv>\n";
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];

    auto puzzles = load_puzzles(in_path);
    int N = (int)puzzles.size();
    std::vector<std::vector<Inst>> solutions(N);
    std::atomic<int> done{0};
    std::atomic<long long> total_steps{0};

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < N; ++i) {
        solutions[i] = solve_puzzle(puzzles[i]);
        std::string err;
        if (!verify_program(puzzles[i], solutions[i], &err)) {
            #pragma omp critical
            {
                std::cerr << "VERIFY FAILED on " << puzzles[i].id << ": " << err << "\n";
                std::exit(2);
            }
        }
        total_steps += (long long)solutions[i].size();
        int d = ++done;
        if (d % 100 == 0 || d == N) {
            #pragma omp critical
            {
                std::cerr << "Solved " << d << "/" << N
                          << " (running total=" << total_steps.load() << ")\n";
            }
        }
    }

    // Family breakdown if available
    std::map<std::string, std::pair<int, long long>> fam_stats;
    for (int i = 0; i < N; ++i) {
        std::string fam = puzzles[i].family.empty() ? "all" : puzzles[i].family;
        fam_stats[fam].first++;
        fam_stats[fam].second += (long long)solutions[i].size();
    }
    std::cerr << "=== TOTAL SCORE: " << total_steps.load()
              << " (avg " << (double)total_steps.load() / N << " / puzzle) ===\n";
    for (const auto &[fam, st] : fam_stats) {
        std::cerr << "  " << fam << ": n=" << st.first
                  << " sum=" << st.second
                  << " avg=" << (double)st.second / st.first << "\n";
    }

    std::ofstream out(out_path);
    out << "id,program\n";
    for (int i = 0; i < N; ++i) {
        std::string js = program_to_json(solutions[i]);
        // CSV-escape JSON string (double quotes -> "")
        out << puzzles[i].id << ",\"";
        for (char c : js) {
            if (c == '"') out << "\"\"";
            else out << c;
        }
        out << "\"\n";
    }
    out.close();
    std::cerr << "Wrote " << out_path << "\n";
    return 0;
}
