from sage.all import *
import random
import os
import time

TOY_MODE = False
FLATTER_PATH = "/Users/tafka4/flatter/build-clang/bin/flatter"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def get_params():
    if TOY_MODE:
        return 101, 2048, 80
    else:
        return 509, 2048, 425

def center_lift(x, q):
    x = int(x)
    if x > q // 2: x -= q
    return x

def random_ternary(N):
    return vector(ZZ, [random.choice([-1, 0, 1]) for _ in range(N)])

def conv_cyclic_mod(a, b, q):
    N = len(a)
    res = [0]*N
    for i in range(N):
        ai = int(a[i])
        if ai == 0: continue
        for j in range(N):
            res[(i + j) % N] = (res[(i + j) % N] + ai * int(b[j])) % q
    return vector(ZZ, res)

def build_paper_lattice(A_mat, Tk_vec, q):
    log("  [Step 1] Building Lattice B_k (Eq 3.6)...")
    
    rows_k, cols_n = A_mat.nrows(), A_mat.ncols() # k x N
    N = cols_n
    k = rows_k
    
    N1 = 9
    N2 = q**4
    
    lattice_rows = []
    
    # A_trans = A_mat.transpose() # N x k
    
    for i in range(N):
        row = [0] * (N + 1 + k)
        row[i] = 1 # Identity
        row[N] = 0 # Middle zero
        
        col_vals = A_mat.column(i)
        for j in range(k):
            row[N + 1 + j] = int(col_vals[j]) * N2
            
        lattice_rows.append(row)
        
    row_mid = [0] * (N + 1 + k)
    row_mid[N] = N1
    for j in range(k):
        # Tk_vec[j] is the constant term
        val = int(Tk_vec[j])
        row_mid[N + 1 + j] = -val * N2
    lattice_rows.append(row_mid)
    
    for i in range(k):
        row = [0] * (N + 1 + k)
        row[N + 1 + i] = N2 * q
        lattice_rows.append(row)
        
    B = matrix(ZZ, lattice_rows)
    log(f"    -> Lattice Dimension: {B.nrows()}x{B.ncols()}")
    log(f"    -> Using N1={N1}, N2=q^4")
    
    return B, N1

def call_flatter(B, trial_idx):
    log("  [Step 2] Running Flatter...")
    t_start = time.time()
    
    lat_file = f"fast_{trial_idx}.lat"
    red_file = f"fast_{trial_idx}_red.lat"
    
    with open(lat_file, "w") as f:
        f.write("[")
        for row in B:
            f.write("[" + " ".join(str(x) for x in row) + "]\n")
        f.write("]\n")

    cmd = f"{FLATTER_PATH} {lat_file} {red_file} > /dev/null 2>&1"
    ret = os.system(cmd)
    
    B_red = None
    if ret == 0 and os.path.exists(red_file):
        try:
            rows = []
            with open(red_file, "r") as f:
                content = f.read().replace("[", "").replace("]", "").strip()
                for line in content.splitlines():
                    if line.strip():
                        rows.append([int(x) for x in line.split()])
            if rows:
                B_red = matrix(ZZ, rows)
        except: pass
    
    os.system(f"rm {lat_file} {red_file} 2>/dev/null")
    if B_red:
        log(f"    -> Flatter done in {time.time()-t_start:.2f}s")
    return B_red

def recover_from_basis(B_red, N, N1):
    dim = B_red.nrows()
    
    for i in range(dim):
        row = B_red.row(i)
        val = row[N]
        
        if val == 0: continue
        
        if val % N1 == 0:
            quotient = val // N1
            
            if abs(quotient) != 1: 
                continue
                
            r_cand = []
            possible = True
            for x in row[:N]:
                if x % quotient != 0:
                    possible = False
                    break
                r_cand.append(x // quotient)
            
            if not possible: continue
            
            r_vec = vector(ZZ, r_cand)
            
            if all(x in [-1, 0, 1] for x in r_vec):
                log(f"    [FOUND] Candidate found at row {i}!")
                return r_vec
                
    return None

def run_attack(trial_idx):
    N, q, k = get_params()
    log(f"--- Trial {trial_idx} (N={N}, k={k}) ---")

    r_true = random_ternary(N)
    m_true = random_ternary(N)
    h = vector(ZZ, [ZZ.random_element(0, q) for _ in range(N)])
    
    hr = conv_cyclic_mod(h, r_true, q)
    c = vector(ZZ, [ (3*int(hr[i]) + int(m_true[i])) % q for i in range(N) ])
    
    inv3 = inverse_mod(3, q)
    
    Tk_vals = []
    for i in range(k):
        # Using known message parts
        val = (int(c[i]) - int(m_true[i])) * int(inv3) % q
        Tk_vals.append(val)
    Tk_vec = vector(ZZ, Tk_vals)
    
    A_rows = []
    for i in range(k):
        row = [0]*N
        for j in range(N):
            row[j] = h[(i - j) % N]
        A_rows.append(row)
    A_mat = matrix(ZZ, A_rows)
    
    Bk, N1 = build_paper_lattice(A_mat, Tk_vec, q)
    
    B_red = call_flatter(Bk, trial_idx)
    if B_red is None: return False
    
    r_rec = recover_from_basis(B_red, N, N1)
    
    if r_rec is not None:
        if r_rec == r_true:
            log("  [Success] Recovered r exactly!")
            return True
        elif r_rec == -r_true:
            log("  [Success] Recovered -r (valid)!")
            return True
        else:
            log("  [Fail] Recovered ternary vector but incorrect.")
            return False
            
    log("  [Fail] No valid r candidate found.")
    return False

if __name__ == "__main__":
    if run_attack(0):
        print(">>> ATTACK SUCCESS <<<")
    else:
        print(">>> ATTACK FAILED <<<")