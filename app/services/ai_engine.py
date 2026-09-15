import asyncio
import time
import numpy as np

def _calculate_heavy_tactical_simulation(data: dict) -> dict:
    """
    Función síncrona con cálculo pesado (CPU-bound).
    Esta función NO corre en el hilo principal.
    """
    # Simulamos un procesamiento matricial pesado con Numpy
    time.sleep(0.5)  # Simulación de tiempo de cómputo
    matrix = np.random.rand(100, 100)
    result = float(np.mean(matrix))
    
    return {
        "success_probability": round(result, 4),
        "recommended_action": "PASS_TO_WING",
        "heat_map_coords": [[12.5, 45.2], [30.1, 50.4]]
    }

class AIEngineService:
    async def process_simulation_async(self, simulation_data: dict) -> dict:
        """
        Delega el cálculo pesado a un HILO SECUNDARIO (Thread Pool)
        liberando por completo el hilo principal de FastAPI.
        """
        return await asyncio.to_thread(_calculate_heavy_tactical_simulation, simulation_data)

ai_engine_service = AIEngineService()