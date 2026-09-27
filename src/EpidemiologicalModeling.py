import matplotlib.pyplot as plt


# =============================================================
# Definición de funciones para el modelo SEIR
# =============================================================


# Calcula las derivadas del sistema SEIR para un instante t
# Retorna las derivadas dS/dt, dE/dt, dI/dt, dR/dt
def derivativesSeir(t, state, beta, sigma, gamma, N):

    S, E, I, R = state

    dSdt = -(beta * S * I) / N
    dEdt = (beta * S * I) / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    return [dSdt, dEdt, dIdt, dRdt]

# Implementación manual del método Runge-Kutta de 4to Orden (RK4)
# Retorna una lista con el estado del sistema en el siguiente paso [S, E, I, R].
def rk4(t, state, h, beta, sigma, gamma, N):
    
    # k1 = f(t, y)
    k1 = derivativesSeir(t, state, beta, sigma, gamma, N)

    # state_k2 = y + (h/2)*k1
    state_k2 = [s + (h / 2.0) * k for s, k in zip(state, k1)]
    # k2 = f(t + h/2, y + (h/2)*k1)
    k2 = derivativesSeir(t + h / 2.0, state_k2, beta, sigma, gamma, N)

    # state_k3 = y + (h/2)*k2
    state_k3 = [s + (h / 2.0) * k for s, k in zip(state, k2)]
    # k3 = f(t + h/2, y + (h/2)*k2)
    k3 = derivativesSeir(t + h / 2.0, state_k3, beta, sigma, gamma, N)

    # state_k4 = y + h*k3
    state_k4 = [s + h * k for s, k in zip(state, k3)]
    # k4 = f(t + h, y + h*k3)
    k4 = derivativesSeir(t + h, state_k4, beta, sigma, gamma, N)

    # y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    next_state = [
        s + (h / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    ]

    return next_state


# Simula el modelo SEIR para un escenario dado (A o B)
# Retorna una lista de tiempos y una lista de estados (S, E, I, R) para cada paso de tiempo.
def simulateSeir(
    scenario_type,
    N,
    S0,
    E0,
    I0,
    R0,
    sigma,
    gamma,
    t_max,
    h,
    beta_initial,
    beta_intervention=None,
    t_intervention=30,
):

    num_steps = int(t_max / h)

    times = [0.0]
    states = [[S0, E0, I0, R0]]

    current_state = [S0, E0, I0, R0]
    t = 0.0

    for _ in range(num_steps):
        # Determinar el valor de beta según el escenario
        if scenario_type == "B" and t >= t_intervention:
            beta = beta_intervention
        else:
            beta = beta_initial

        current_state = rk4(t, current_state, h, beta, sigma, gamma, N)
        t += h

        times.append(t)
        states.append(current_state)

    return times, states


# ========================================================
# Definición de parámetros y condiciones iniciales
# ========================================================

N = 3e7  # 30,000,000
I0 = 100.0
E0 = 0.0
R0 = 0.0
S0 = N - I0

sigma = 0.2
gamma = 0.1
beta_initial = 0.6
beta_intervention = 0.25
t_max = 200.0  # Días de simulación propuestos para el estudio
h = 0.1  # Tamaño del paso (equivale a dividir cada día en 10 intervalos de 2.4 horas)

# ==========================================================
# Simulación de los escenarios A y B
# ==========================================================

# Escenario A
times_A, states_A = simulateSeir(
    scenario_type="A",
    N=N,
    S0=S0,
    E0=E0,
    I0=I0,
    R0=R0,
    sigma=sigma,
    gamma=gamma,
    t_max=t_max,
    h=h,
    beta_initial=beta_initial
)

# Escenario B
times_B, states_B = simulateSeir(
    scenario_type="B",
    N=N,
    S0=S0,
    E0=E0,
    I0=I0,
    R0=R0,
    sigma=sigma,
    gamma=gamma,
    t_max=t_max,
    h=h,
    beta_initial=beta_initial,
    beta_intervention=beta_intervention,
)

# ==========================================================
# Gráficas
# ==========================================================
S_A, E_A, I_A, R_A = zip(*states_A)
S_B, E_B, I_B, R_B = zip(*states_B)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Escenario A
axes[0].plot(times_A, S_A, label="Susceptibles (S)", color="blue")
axes[0].plot(times_A, E_A, label="Expuestos (E)", color="orange")
axes[0].plot(times_A, I_A, label="Infectados (I)", color="red")
axes[0].plot(times_A, R_A, label="Recuperados (R)", color="green")
axes[0].set_title("Escenario A: Libre (β = 0.6)")
axes[0].set_xlabel("Tiempo (días)")
axes[0].set_ylabel("Población")
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend()

# Escenario B
axes[1].plot(times_B, S_B, label="Susceptibles (S)", color="blue")
axes[1].plot(times_B, E_B, label="Expuestos (E)", color="orange")
axes[1].plot(times_B, I_B, label="Infectados (I)", color="red")
axes[1].plot(times_B, R_B, label="Recuperados (R)", color="green")
axes[1].axvline(
    x=30,
    color="black",
    linestyle=":",
    linewidth=2,
    label="Intervención (t=30, β=0.25)",
)
axes[1].set_title("Escenario B: Intervención (β reduce a 0.25 en t=30)")
axes[1].set_xlabel("Tiempo (días)")
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.show()

# ======================================================================
# Análisis de umbral (R0) y picos de infectados
# ======================================================================
print("==========================================")
print("          ANÁLISIS DE UMBRAL (R0)         ")
print("==========================================")

# Cálculo numérico de R0 = beta / gamma
R0_A = beta_initial / gamma
R0_B_initial = beta_initial / gamma
R0_B_intervention = beta_intervention / gamma

# Extracción de picos de infectados
I_A_max = max(I_A)
peak_time_A = times_A[I_A.index(I_A_max)]

I_B_max = max(I_B)
peak_time_B = times_B[I_B.index(I_B_max)]

print(f"Escenario A (Libre):")
print(f"  - R0 = {R0_A:.2f}")
print(f"  - Pico máximo de infectados: {I_A_max:,.0f} personas")
print(f"  - Día del pico: {peak_time_A:.1f} días\n")

print(f"Escenario B (Intervención t >= 30):")
print(f"  - R0 inicial (t < 30): {R0_B_initial:.2f}")
print(f"  - R0 tras intervención (t >= 30): {R0_B_intervention:.2f}")
print(f"  - Pico máximo de infectados: {I_B_max:,.0f} personas")
print(f"  - Día del pico: {peak_time_B:.1f} días")

# ==========================================================
# Demostración de Conservación de la Población
# ==========================================================
max_error_A = max([abs(sum(state) - N) for state in states_A])
max_error_B = max([abs(sum(state) - N) for state in states_B])

print("==========================================")
print(" ANÁLISIS DE CONSERVACIÓN DE LA POBLACIÓN ")
print("==========================================")
print("Se espera que la suma de S, E, I y R sea igual a N en todo momento.")
print(f"Escenario A: Error máximo de conservación: {max_error_A:.2e} personas (Aproximado a 2 decimales: {max_error_A:.2f})")
print(f"Escenario B: Error máximo de conservación: {max_error_B:.2e} personas (Aproximado a 2 decimales: {max_error_B:.2f})")