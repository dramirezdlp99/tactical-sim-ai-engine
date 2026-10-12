"""
monte_carlo.py

Brecha #2 del Caso de Estudio 2 (seccion 2.2): "Calculo Multihilo en
Memoria Compartida: Ejecuta simulaciones numericas de trayectorias sobre
miles de iteraciones posibles. Al utilizar estructuras de memoria
compartida entre procesos, los calculos pesados leen y escriben datos en
la misma memoria... sin demoras por transferencia de informacion."

ANTES: _run_simulation_sync (ai_engine.py) calculaba la jugada UNA sola
vez con formulas geometricas deterministas, delegado a un solo hilo
secundario (asyncio.to_thread). Eso es real (no bloquea el event loop de
FastAPI), pero no son "miles de iteraciones" ni usa "memoria compartida
entre procesos" como pide el caso de estudio -- y un solo hilo de Python
nunca ejecuta CPU en paralelo de verdad, por el GIL (Global Interpreter
Lock: solo un hilo de Python corre bytecode a la vez, sin importar
cuantos hilos se abran).

AHORA: este modulo ejecuta una simulacion Monte Carlo real. La posicion
de cada jugador tiene una pequena incertidumbre de ejecucion (un
defensor no siempre reacciona exactamente igual, un tirador no siempre
suelta el balon en el mismo punto exacto, el viento afecta un saque de
tenis). Se generan miles de variantes aleatorias de la jugada (por
defecto 4000) y se reparten entre varios PROCESOS reales del sistema
operativo (modulo multiprocessing, NO threading -- un proceso si evade
el GIL, porque cada uno tiene su propio interprete de Python corriendo
en un nucleo de CPU distinto). Todos los procesos escriben sus
resultados directamente en el MISMO bloque de memoria compartida
(multiprocessing.shared_memory), para no tener que serializar y devolver
miles de numeros al proceso principal por un pipe -- exactamente la
transferencia de informacion que el caso de estudio pide evitar.

Al final, el proceso principal lee ese bloque compartido como una lista
de numeros y calcula la probabilidad de exito promedio real (la media de
miles de intentos con variaciones realistas) y su desviacion estandar
(que tan consistente es la jugada bajo esas variaciones).

Guardar en: tactical-sim-ai-engine/app/services/monte_carlo.py
"""

import os
import random
import struct
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from typing import Callable, Dict, Optional

# Numero de iteraciones Monte Carlo por simulacion solicitada. Configurable
# por variable de entorno para poder bajarlo en un servidor gratuito con
# poca CPU (Render free tier) sin tocar codigo -- nada de esto tiene costo,
# es CPU del mismo servicio que ya esta desplegado.
MONTE_CARLO_ITERATIONS = int(os.getenv("MONTE_CARLO_ITERATIONS", "4000"))

# Numero de procesos del sistema operativo que se reparten el trabajo. Se
# limita a un maximo razonable para no saturar un servidor gratuito de CPU
# compartida; nunca se usan mas procesos que nucleos visibles ni que un
# limite duro definido por variable de entorno.
MONTE_CARLO_WORKERS = max(1, min(int(os.getenv("MONTE_CARLO_WORKERS", "4")), os.cpu_count() or 1))

# Desviacion estandar de la incertidumbre de ejecucion, en las mismas
# unidades 0-100 que usa el frontend para las coordenadas de la cancha.
# Representa cuanto puede variar la posicion real de un jugador respecto a
# donde el entrenador lo dibujo (reaccion del defensor, pulso del
# tirador, etc.). Es "ruido de ejecucion" pequeno a proposito -- no un
# jugador reposicionandose por completo.
POSITION_NOISE_STD = float(os.getenv("MONTE_CARLO_NOISE_STD", "2.5"))

# Cada slot en la memoria compartida guarda un double de Python (8 bytes).
_DOUBLE_SIZE = struct.calcsize("d")

# Pool de procesos PERSISTENTE: se crea una sola vez (la primera vez que
# se llama run_monte_carlo) y se reutiliza en todas las peticiones
# siguientes. Crear procesos del sistema operativo no es gratis en tiempo
# (toma decenas de milisegundos); si se creara un pool nuevo en cada
# peticion HTTP, cada simulacion pagaria ese costo de arranque. Con un
# pool persistente, ese costo se paga una sola vez por el tiempo de vida
# del servicio.
_pool: Optional[ProcessPoolExecutor] = None
_pool_lock = threading.Lock()


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=MONTE_CARLO_WORKERS)
    return _pool


def _clamp0_100(value: float) -> float:
    return max(0.0, min(100.0, value))


def _jitter_positions(payload: Dict, rng: random.Random) -> Dict:
    """Devuelve una copia del payload con una pequena variacion aleatoria
    (ruido gaussiano) en cada coordenada, simulando la incertidumbre real
    de ejecucion de una jugada."""

    def jitter_point(p: Dict) -> Dict:
        return {
            **p,
            "x": _clamp0_100(p["x"] + rng.gauss(0, POSITION_NOISE_STD)),
            "y": _clamp0_100(p["y"] + rng.gauss(0, POSITION_NOISE_STD)),
        }

    return {
        "match_id": payload.get("match_id"),
        "team_home_positions": [jitter_point(p) for p in payload["team_home_positions"]],
        "team_away_positions": [jitter_point(p) for p in payload["team_away_positions"]],
        "ball_position": jitter_point(payload["ball_position"]),
    }


def _worker_run_chunk(
    shm_name: str,
    start_idx: int,
    end_idx: int,
    positions_payload: Dict,
    deterministic_fn_name: str,
    seed: int,
) -> None:
    """
    Corre en un PROCESO hijo real del sistema operativo (no un hilo): su
    propio interprete de Python, su propia memoria privada -- EXCEPTO el
    bloque de memoria compartida, que abre por nombre. Calcula su porcion
    (start_idx:end_idx) de las iteraciones Monte Carlo y escribe cada
    resultado directamente en esa memoria compartida.
    """
    # Import diferido a proposito: evita import circular con ai_engine.py
    # (que importa run_monte_carlo de este modulo) y asegura que, si el
    # sistema operativo usa el metodo "spawn" en vez de "fork", el
    # proceso hijo pueda resolver la funcion determinista por su nombre.
    from app.services import ai_engine as engine_module

    # Si el motor ofrece una version ligera "<nombre>_prob_only" (solo calcula
    # success_probability, sin mapa de calor ni notas), se usa esa: aqui solo se
    # lee success_probability, asi que el resto seria trabajo tirado 4000 veces.
    # Si no existe, se usa la funcion completa de siempre.
    deterministic_fn: Callable[[Dict], Dict] = (
        getattr(engine_module, deterministic_fn_name + "_prob_only", None)
        or getattr(engine_module, deterministic_fn_name)
    )

    # Cada proceso hijo necesita su propia semilla para no repetir
    # exactamente la misma secuencia "aleatoria" que sus hermanos.
    rng = random.Random(seed)

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        for i in range(start_idx, end_idx):
            noisy_payload = _jitter_positions(positions_payload, rng)
            result = deterministic_fn(noisy_payload)
            success = float(result.get("success_probability", 0.0))
            offset = i * _DOUBLE_SIZE
            shm.buf[offset: offset + _DOUBLE_SIZE] = struct.pack("d", success)
    finally:
        # Este proceso deja de usar la memoria compartida; NO la destruye
        # (unlink lo hace unicamente el proceso principal, una vez leidos
        # todos los resultados).
        shm.close()


def run_monte_carlo(positions_payload: Dict, deterministic_fn_name: str) -> Dict:
    """
    Orquesta la simulacion multiproceso completa:

    1. Reserva un bloque de memoria compartida del tamano exacto para
       MONTE_CARLO_ITERATIONS resultados (un double de 8 bytes cada uno).
    2. Reparte los indices [0, MONTE_CARLO_ITERATIONS) en
       MONTE_CARLO_WORKERS tramos contiguos, uno por proceso.
    3. Usa el pool de procesos persistente para que cada tramo se calcule
       en paralelo, escribiendo cada proceso directamente en su porcion
       del bloque compartido.
    4. Una vez que todos los procesos terminan, el proceso principal lee
       el bloque compartido completo y calcula la media (probabilidad de
       exito real, promediada sobre miles de intentos) y la desviacion
       estandar (consistencia de la jugada bajo variaciones realistas).
    5. Libera (unlink) la memoria compartida de ESTA peticion -- cada
       peticion HTTP reserva su propio bloque con nombre unico generado
       por el sistema operativo, asi que varias simulaciones concurrentes
       nunca comparten ni pisan el mismo bloque.
    """
    total = MONTE_CARLO_ITERATIONS
    shm = shared_memory.SharedMemory(create=True, size=total * _DOUBLE_SIZE)
    try:
        chunk_size = max(1, total // MONTE_CARLO_WORKERS)
        chunks = []
        idx = 0
        while idx < total:
            end = min(idx + chunk_size, total)
            chunks.append((idx, end))
            idx = end

        pool = _get_pool()
        futures = [
            pool.submit(
                _worker_run_chunk,
                shm.name,
                start,
                end,
                positions_payload,
                deterministic_fn_name,
                1000 + worker_idx,  # semilla distinta por proceso
            )
            for worker_idx, (start, end) in enumerate(chunks)
        ]
        for f in futures:
            f.result()  # propaga cualquier excepcion real ocurrida en un proceso hijo

        # Lee el bloque compartido completo: cada proceso hijo escribio
        # directamente aqui, no hubo que devolver los datos por un pipe.
        raw = bytes(shm.buf[: total * _DOUBLE_SIZE])
        samples = struct.unpack(f"{total}d", raw)

        mean = sum(samples) / total
        variance = sum((s - mean) ** 2 for s in samples) / total
        std_dev = variance ** 0.5

        return {
            "success_probability_mc": round(mean, 4),
            "success_std_dev": round(std_dev, 4),
            "iterations": total,
            "workers": len(chunks),
        }
    finally:
        shm.close()
        shm.unlink()  # el proceso principal es el dueño del bloque: libera la memoria del SO
