import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem
import ufl

# Importa la funzione principale dal tuo script
from incompressible import solve_model

def run_convergence_test():
    # 1. Definisci i livelli di infittimento della mesh
    # h_factor piccolo = mesh fitta.
    # Usiamo 0.35 per la Reference (abbastanza fitta) e valori decrescenti per i test.
    # (Attenzione: se la RAM va in crisi, aumenta leggermente questi valori, es. h_ref = 0.5)
    h_ref_val = 0.35
    h_factors = [1.0, 0.8, 0.6, 0.45]
    
    errors_L2 = []
    element_sizes = []

    print(f"\n========================================================")
    print(f" 1. Calcolo Soluzione di Riferimento (h_factor = {h_ref_val})")
    print(f"========================================================\n")
    domain_ref, W_ref, U_ref = solve_model(h_factor=h_ref_val, run_postprocessing=False)
    
    # ... (dopo aver calcolato U_ref) ...
    
    # 1. Estraiamo tutti e tre i campi di riferimento
    V_u_ref, dofs_u_ref = W_ref.sub(0).collapse()
    V_phi_ref, dofs_phi_ref = W_ref.sub(1).collapse()
    V_p_ref, dofs_p_ref = W_ref.sub(2).collapse()

    u_ref = fem.Function(V_u_ref)
    u_ref.x.array[:] = U_ref.x.array[dofs_u_ref]
    phi_ref = fem.Function(V_phi_ref)
    phi_ref.x.array[:] = U_ref.x.array[dofs_phi_ref]
    p_ref = fem.Function(V_p_ref)
    p_ref.x.array[:] = U_ref.x.array[dofs_p_ref]

    # Prepariamo le liste per i 3 errori
    errs_u, errs_phi, errs_p, element_sizes = [], [], [], []
    dx_ref = ufl.Measure("dx", domain=domain_ref)
    
    # Setup per l'interpolazione (celle della mesh fine)
    map_c = domain_ref.topology.index_map(domain_ref.topology.dim)
    cells = np.arange(map_c.size_local + map_c.num_ghosts, dtype=np.int32)

    for h in h_factors:
        print(f">>> Risoluzione per h_factor = {h}...")
        domain_h, W_h, U_h = solve_model(h_factor=h, run_postprocessing=False)
        
        # 2. Estraiamo i campi grossolani
        V_u_h, dofs_u_h = W_h.sub(0).collapse()
        V_phi_h, dofs_phi_h = W_h.sub(1).collapse()
        V_p_h, dofs_p_h = W_h.sub(2).collapse()
        
        u_h = fem.Function(V_u_h)
        u_h.x.array[:] = U_h.x.array[dofs_u_h]
        phi_h = fem.Function(V_phi_h)
        phi_h.x.array[:] = U_h.x.array[dofs_phi_h]
        p_h = fem.Function(V_p_h)
        p_h.x.array[:] = U_h.x.array[dofs_p_h]
        
        # 3. Interpoliamo tutti e tre i campi sulla mesh di riferimento
        u_h_interp = fem.Function(V_u_ref)
        nmm_u = fem.create_interpolation_data(V_u_ref, V_u_h, cells, padding=1e-12)
        u_h_interp.interpolate_nonmatching(u_h, cells, interpolation_data=nmm_u)

        phi_h_interp = fem.Function(V_phi_ref)
        nmm_phi = fem.create_interpolation_data(V_phi_ref, V_phi_h, cells, padding=1e-12)
        phi_h_interp.interpolate_nonmatching(phi_h, cells, interpolation_data=nmm_phi)

        p_h_interp = fem.Function(V_p_ref)
        nmm_p = fem.create_interpolation_data(V_p_ref, V_p_h, cells, padding=1e-12)
        p_h_interp.interpolate_nonmatching(p_h, cells, interpolation_data=nmm_p)

        # 4. Calcoliamo i tre errori separatamente
        def calc_relative_L2(f_ref, f_interp):
            err_form = fem.form(ufl.inner(f_ref - f_interp, f_ref - f_interp) * dx_ref)
            err_abs = np.sqrt(domain_ref.comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

            norm_form = fem.form(ufl.inner(f_ref, f_ref) * dx_ref)
            norm_ref = np.sqrt(domain_ref.comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))

            if norm_ref < 1e-14:
                return err_abs

            return err_abs / norm_ref

        e_u = calc_relative_L2(u_ref, u_h_interp)
        e_phi = calc_relative_L2(phi_ref, phi_h_interp)
        e_p = calc_relative_L2(p_ref, p_h_interp)
        
        errs_u.append(e_u)
        errs_phi.append(e_phi)
        errs_p.append(e_p)
        element_sizes.append(h)
        print(f"    --> Errori: u={e_u:.2e}, phi={e_phi:.2e}, p={e_p:.2e}")

    # ========================================================
    # GRAFICO TRIPLO
    # ========================================================
    if domain_ref.comm.rank == 0:
        log_h = np.log(element_sizes)
        # Calcolo pendenze
        p_u, _ = np.polyfit(log_h, np.log(errs_u), 1)
        p_phi, _ = np.polyfit(log_h, np.log(errs_phi), 1)
        p_pres, _ = np.polyfit(log_h, np.log(errs_p), 1)
        
        plt.figure(figsize=(9, 7))
        plt.loglog(element_sizes, errs_u, 'o-', color="#d62728", lw=2.5, label=f"Spostamento $u$ ($p \\approx {p_u:.2f}$)")
        plt.loglog(element_sizes, errs_phi, 's-', color="#1f77b4", lw=2.5, label=f"Potenziale $\\phi$ ($p \\approx {p_phi:.2f}$)")
        plt.loglog(element_sizes, errs_p, '^-', color="#2ca02c", lw=2.5, label=f"Pressione $p$ ($p \\approx {p_pres:.2f}$)")
        
        plt.xlabel("Dimensione relativa della mesh ($h\\_factor$)", fontsize=12)
        plt.ylabel("Errore $L^2$ rispetto alla Reference", fontsize=12)
        plt.title("Convergenza Spaziale - Formulazione Mista Magneto-Elastica", fontsize=14, fontweight="bold")
        plt.grid(True, which="major", ls="-", alpha=0.6)
        plt.grid(True, which="minor", ls="--", alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig("grafico_convergenza_misto.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    run_convergence_test()