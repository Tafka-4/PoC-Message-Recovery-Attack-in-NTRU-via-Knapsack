from sage.all import *
import random
import os
import time

FLATTER_PATH = "/Users/tafka4/flatter/build-clang/bin/flatter"

N, q = 509, 2048
N1 = 9

K1_MSG = 300 
K2_NONCE = 180 

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

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

def call_flatter(B):
    lat_file, red_file = "hybrid.lat", "hybrid_red.lat"
    with open(lat_file, "w") as f:
        f.write("["); [f.write("["+" ".join(str(x) for x in r)+"]\n") for r in B]; f.write("]\n")
    
    cmd = f"{FLATTER_PATH} {lat_file} {red_file} > /dev/null 2>&1"
    os.system(cmd)
    
    B_red = None
    if os.path.exists(red_file):
        try:
            with open(red_file) as f:
                rows = [[int(x) for x in line.split()] for line in f.read().replace("[","").replace("]","").splitlines() if line.strip()]
            if rows: B_red = matrix(ZZ, rows)
        except: pass
    os.system(f"rm {lat_file} {red_file} 2>/dev/null")
    return B_red

def build_alternative_lattice(A_z, Tz_vec, q, N1):
    rows_k1, cols_unknown = A_z.nrows(), A_z.ncols()
    N2 = q**4
    
    lattice_rows = []
    
    for i in range(cols_unknown):
        row = [0] * (cols_unknown + 1 + rows_k1)
        row[i] = 1
        row[cols_unknown] = 0 # Embedding slot
        
        col_vals = A_z.column(i)
        for j in range(rows_k1):
            row[cols_unknown + 1 + j] = int(col_vals[j]) * N2
        lattice_rows.append(row)
        
    row_mid = [0] * (cols_unknown + 1 + rows_k1)
    row_mid[cols_unknown] = N1
    for j in range(rows_k1):
        # -N2 * Tz
        row_mid[cols_unknown + 1 + j] = -int(Tz_vec[j]) * N2
    lattice_rows.append(row_mid)
    
    for i in range(rows_k1):
        row = [0] * (cols_unknown + 1 + rows_k1)
        row[cols_unknown + 1 + i] = N2 * q
        lattice_rows.append(row)
        
    return matrix(ZZ, lattice_rows)

def recover_partial_r(B_red, N_unknown, N1):
    dim = B_red.nrows()
    for i in range(dim):
        row = B_red.row(i)
        val = row[N_unknown]
        
        if val == 0: continue
        if val % N1 == 0:
            quot = val // N1
            if abs(quot) != 1: continue
            
            cand = []
            possible = True
            for x in row[:N_unknown]:
                if x % quot != 0:
                    possible = False
                    break
                cand.append(x // quot)
            
            if not possible: continue
            
            if all(x in [-1, 0, 1] for x in cand):
                return vector(ZZ, cand)
    return None

def run_hybrid_attack():
    print(f"=== Alternative Attack (Hybrid Knowledge) ===")
    print(f"Parameters: N={N}, q={q}")
    print(f"Assumptions: Known Msg={K1_MSG}, Known Nonce={K2_NONCE}")
    print(f"Total Known: {K1_MSG + K2_NONCE} / {2*N} (approx {float(((K1_MSG+K2_NONCE)/(2*N))*100):.1f}%)")
    
    m_true = random_ternary(N)
    r_true = random_ternary(N)
    h = vector(ZZ, [ZZ.random_element(0, q) for _ in range(N)])
    
    # Encrypt
    hr = conv_cyclic_mod(h, r_true, q)
    c = vector(ZZ, [ (3*int(hr[i]) + int(m_true[i])) % q for i in range(N) ])
    
    known_r_indices = list(range(K2_NONCE))
    unknown_r_indices = list(range(K2_NONCE, N))
    num_unknown = len(unknown_r_indices)
    
    inv3 = inverse_mod(3, q)
    Tk1_vals = []
    for i in range(K1_MSG):
        val = (int(c[i]) - int(m_true[i])) * int(inv3) % q
        Tk1_vals.append(val)
    Tk1_vec = vector(ZZ, Tk1_vals)
    
    A_rows = []
    for i in range(K1_MSG):
        row = [0]*N
        for j in range(N):
            row[j] = h[(i - j) % N]
        A_rows.append(row)
    A_full = matrix(ZZ, A_rows)
    
    print("[Step 1] Adjusting Target Vector with known Nonce...")
    S_vec = vector(ZZ, [0]*K1_MSG)
    for idx in known_r_indices:
        r_val = int(r_true[idx])
        if r_val == 0: continue
        
        col = A_full.column(idx)
        S_vec += r_val * col
        
    Tz_vec = vector(ZZ, [(Tk1_vec[i] - S_vec[i]) % q for i in range(K1_MSG)])
    
    print(f"[Step 2] Reducing Matrix Dimension (Removing {K2_NONCE} columns)...")
    Az = A_full[:, K2_NONCE:N]
    
    print(f"    -> Reduced Matrix Size: {Az.nrows()} x {Az.ncols()}")
    print("[Step 3] Building Alternative Lattice...")
    Bz = build_alternative_lattice(Az, Tz_vec, q, N1)
    
    print("[Step 4] Running Flatter...")
    t0 = time.time()
    B_red = call_flatter(Bz)
    if B_red is None:
        print("Flatter Failed.")
        return
        
    print(f"    -> Reduction time: {time.time()-t0:.2f}s")
    r_partial_rec = recover_partial_r(B_red, num_unknown, N1)
    
    if r_partial_rec:
        print("[Step 5] Partial Vector Found! Reconstructing full r...")
        
        r_constructed = [0]*N
        
        for idx in known_r_indices:
            r_constructed[idx] = int(r_true[idx])
            
        candidates = [r_partial_rec, -r_partial_rec]
        
        for cand in candidates:
            temp_r = list(r_constructed) # Copy
            for i, val in enumerate(cand):
                target_idx = unknown_r_indices[i]
                temp_r[target_idx] = val
            
            vec_r = vector(ZZ, temp_r)
            
            if vec_r == r_true:
                print("\n>>> ATTACK SUCCESS: Full Nonce Recovered! <<<")
                return
            
            hr_test = conv_cyclic_mod(h, vec_r, q)
            m_test = []
            valid_m = True
            for i in range(N):
                val = (int(c[i]) - 3*int(hr_test[i])) % q
                if val > q//2: val -= q
                if val not in [-1,0,1]:
                    valid_m = False
                    break
            
            if valid_m:
                print("\n>>> ATTACK SUCCESS: Valid Message Decrypted! <<<")
                # print(f"Recovered r: {vec_r[:10]}...")
                return

    print("\n>>> ATTACK FAILED: Could not recover remaining parts. <<<")

if __name__ == "__main__":
    run_hybrid_attack()