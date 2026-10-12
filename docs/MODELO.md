<!-- Guardar en: tactical-sim-ai-engine/docs/MODELO.md -->
# Modelo Predictivo de Evaluación Espacial y "Cuerpo Técnico Rival"

Documento para el dueño del proyecto. Está pensado para poder explicárselo a un profesor sin ser experto en ML.

## 1. ¿Qué es?

El AI Engine ahora tiene dos piezas nuevas que trabajan juntas:

1. **Modelo predictivo (regresión logística).** Recibe una descripción numérica de la jugada (por ejemplo "el defensor más cercano está a 3 pies del tirador, el tirador está a 12 pies del aro") y devuelve una probabilidad entre 0 y 1. Una regresión logística es el modelo de clasificación más básico y explicable: suma cada característica multiplicada por un peso aprendido y convierte el total en probabilidad. Los pesos se pueden leer en `app/ml/models/*.json`.
2. **Cuerpo Técnico Rival** (`app/services/defense_response.py`). Es un conjunto de reglas de defensa (no entrenado) que predice dónde estarían los defensores *después de reaccionar* a la jugada. El modelo evalúa la jugada contra esa defensa reajustada, no contra la defensa estática que dibujó el entrenador.

Flujo completo de una petición:

```
posiciones dibujadas -> reacción defensiva -> características -> modelo -> probabilidad
                                                                        -> promedio de 4000 variantes Monte Carlo
```

La acción recomendada, la nota táctica y el mapa de calor siguen saliendo de la geometría de siempre. El modelo aporta la probabilidad de éxito, el tiro limpio y la efectividad del pase.

## 2. ¿Qué predice cada "cabeza"?

Cada pregunta distinta tiene su propia regresión logística (se llama "cabeza"):

| Deporte | Cabeza | Pregunta que responde |
|---|---|---|
| Baloncesto | `shot` (tiro) | ¿Anota el portador si tira ahora? (campo `clean_shot_probability`) |
| Baloncesto | `pass` (pase) | ¿Se completa el pase a un compañero? (campo `pass_effectiveness`) |
| Baloncesto | `drive` (penetración) | ¿La penetración termina en canasta o falta a favor? |
| Tenis | `serve` (saque) | ¿Gana el punto quien saca? |

`success_probability` es la de la acción recomendada. Si la acción es "reposicionar y esperar", es la mejor de las tres opciones con un descuento de 15 % por perder tiempo (decisión de diseño, no un valor medido).

Las características (`app/ml/features.py`) son cosas como distancia al aro en pies, distancia del defensor más cercano, si la línea de pase está bloqueada, cuántos defensores protegen el aro, y en tenis cuántos pies tiene que moverse el restador de lado para llegar a la pelota. Se calculan con **el mismo código** al entrenar y en producción, para evitar que el modelo reciba en producción números calculados distinto a como los vio al entrenar.

## 3. ¿Con qué datos se entrenó?

**Con datos sintéticos.** No existen datos históricos reales en este proyecto (no hay tracking de jugadores ni registro de tiros, pases o saques). Por eso `app/ml/dataset.py` fabrica 80 000 situaciones por deporte con semilla fija (42), de forma reproducible:

1. Se inventan posiciones variadas (defensa pegada, normal, suelta, ayudas en la pintura, defensores "perdidos").
2. Cada situación tiene una **probabilidad verdadera** calculada con reglas de conocimiento general del deporte, cada una con su constante comentada en el código. Por ejemplo: acierto cerca del aro ~62-64 %, media distancia ~40-45 %, triple ~36 %; un defensor a menos de 2 pies resta ~11-14 puntos; un defensor sobre la línea de pase puede interceptarla; en tenis, el tiempo que tarda la pelota en llegar y la velocidad lateral del restador.
3. El resultado de cada intento (anota / no anota) se **sortea** con esa probabilidad (Bernoulli) y se añade un factor oculto de habilidad que el modelo no ve.
4. El modelo se entrena **solo con los resultados 0/1**, nunca con las probabilidades. Tiene que "descubrir" los patrones.

La verdad de terreno no copia las fórmulas que tenía `ai_engine.py` (usa otras variables, curvas por tramos, interacciones y azar), para que el modelo no sea simplemente una copia de las fórmulas viejas.

**Por qué esto es honesto pero limitado:** el modelo aprende los patrones que *nosotros* pusimos en el generador. Demuestra el flujo completo de ML (datos, características, entrenamiento, validación, despliegue, fallback), pero NO demuestra que las probabilidades coincidan con partidos reales. La respuesta incluye `model_info`, que lo dice con la palabra "sintéticas".

## 4. Resultados de validación

Se mide con el 20 % de las situaciones que el modelo no vio. "Baseline" es un predictor tonto que siempre responde la tasa promedio. "Techo" es el AUC que lograría alguien que conociera la probabilidad verdadera del generador; ningún modelo puede pasarlo porque el resultado final es un sorteo.

| Cabeza | AUC | Log-loss (baseline) | Brier (baseline) | ECE (baseline) | Techo AUC |
|---|---|---|---|---|---|
| Baloncesto, tiro | 0.614 | 0.617 (0.637) | 0.214 (0.222) | 0.008 (0.005) | 0.614 |
| Baloncesto, pase | 0.805 | 0.506 (0.652) | 0.170 (0.230) | 0.009 (0.007) | 0.811 |
| Baloncesto, penetración | 0.631 | 0.601 (0.624) | 0.206 (0.216) | 0.008 (0.001) | 0.634 |
| Tenis, saque | 0.635 | 0.618 (0.644) | 0.214 (0.226) | 0.007 (0.006) | 0.641 |

Cómo leerlo:

- **AUC**: probabilidad de que el modelo ponga más alta la probabilidad de un éxito real que la de un fallo real. 0.5 es azar. Un AUC de 0.61 en el tiro es bajo *porque meter o fallar un tiro es muy aleatorio*, incluso conociendo toda la situación; el techo teórico también es 0.614, así que el modelo está aprovechando casi toda la información disponible. El pase tiene más AUC porque es más predecible.
- **Log-loss y Brier**: menores son mejores. El modelo supera al baseline en todas las cabezas.
- **ECE** (calibración): si el modelo dice 70 %, ¿pasa ~70 % de las veces? Valores de 0.007-0.009 son buenos. El baseline tiene ECE bajo por construcción (siempre dice el promedio), así que esa columna no es una comparación justa; solo muestra que el modelo no está descalibrado.

## 5. Cuerpo Técnico Rival (reglas)

Constantes en `defense_response.py`, todas comentadas:

- **Emparejamiento hombre a hombre voraz:** se ordenan todas las parejas defensor-atacante por distancia y se asigna la más cercana libre. Es O(n²) y rápido (se ejecuta 4000 veces por petición).
- **Posición del lado del aro:** el defensor se coloca entre su marca y el aro, a 1.5 pies si marca al balón y 2.5 pies si no.
- **Ayuda en la pintura:** si el balón está a menos de ~15 pies del aro, los 2 defensores cuyas marcas están más lejos del balón se cierran hacia la pintura (a medio camino).
- **Paso máximo:** 10 unidades (~5 pies) por reacción. Nadie se teletransporta.
- **Tenis:** el restador anticipa el saque y se desplaza hacia el punto de bote hasta 4 pies, priorizando el movimiento lateral.

El esquema legible sale en `defense_scheme` (por ejemplo "Hombre a hombre con ayuda en la pintura") y las posiciones predichas en `predicted_defense`.

## 6. Cómo reentrenar

En la máquina del dueño (numpy solo hace falta aquí, no en el servidor):

```bash
pip install numpy
python -m app.ml.train                                  # ambos deportes, 80 000 muestras, semilla 42
python -m app.ml.train --sport tennis --samples 120000 --seed 7
python -m pytest tests -v                               # comprobar que todo sigue pasando
```

Se regeneran `app/ml/models/basketball_v1.json` y `tennis_v1.json` (menos de 10 KB cada uno) y se imprime la tabla de métricas. Basta con hacer commit de los JSON y desplegar. Si se cambia cualquier fórmula de `features.py`, hay que subir `FEATURE_SCHEMA_VERSION` y reentrenar; el servidor se niega a usar pesos con un esquema distinto (usa el fallback).

## 7. Cómo reemplazar los datos sintéticos por reales

Se necesita un CSV por deporte, con coordenadas 0-100 (mismo sistema que el frontend). **Importante:** las posiciones de los defensores deben ser las del momento de la acción (tiro, pase, saque), no las de antes de reaccionar.

**Baloncesto** (una fila por intento; `head` = `shot`, `pass` o `drive`):

| Columna | Significado |
|---|---|
| `head` | `shot`, `pass` o `drive` |
| `outcome` | 1 = éxito (tiro anotado, pase completado, penetración con canasta o falta), 0 = fallo |
| `ball_x`, `ball_y` | posición del balón (el portador es el atacante más cercano) |
| `a1_x`, `a1_y` ... `a5_x`, `a5_y` | atacantes |
| `d1_x`, `d1_y` ... `d5_x`, `d5_y` | defensores en el momento de la acción |
| `receiver` | 1-5, solo para `pass`: número del atacante que recibe (vacío en los demás) |

**Tenis** (una fila por saque):

`outcome` (1 = el sacador ganó el punto), `server_x`, `server_y`, `returner_x`, `returner_y`, `target_x`, `target_y` (punto de bote).

Entrenar:

```bash
python -m app.ml.train --sport basketball --csv partidos_baloncesto.csv
python -m app.ml.train --sport tennis --csv saques_tenis.csv
```

El JSON resultante queda marcado con `"data_source": "real"` y `model_info` dirá "reales" en vez de "sintéticas". Se recomienda al menos unos miles de filas por cabeza.

## 8. Despliegue y seguridad ("el motor nunca se cae")

- En el servidor **no se usa numpy**: la predicción es una suma de productos en Python puro; el JSON se lee una vez por proceso y queda en memoria.
- Si falta el JSON, está corrupto, trae números inválidos, o sus nombres de características no coinciden con `features.py`, o el modelo lanza un error al predecir, el motor vuelve a las fórmulas geométricas anteriores y `model_info` lo indica. Hay pruebas automáticas de esto.
- Monte Carlo usa una versión ligera de la simulación (`_simulate_basketball_prob_only`, `_simulate_tennis_prob_only`) que no calcula mapa de calor ni notas. Las funciones `_simulate_basketball` y `_simulate_tennis` se mantienen con la misma firma.

## 9. Limitaciones honestas

1. **Los datos son sintéticos.** Las probabilidades reflejan nuestros supuestos de dominio, no estadísticas de una liga. Son útiles para comparar jugadas entre sí, no como pronóstico real.
2. **Modelo lineal.** La regresión logística no captura relaciones complejas; se compensa con características cuadráticas, de interacción y "bisagras" elegidas a mano.
3. **El Cuerpo Técnico Rival son reglas, no un modelo entrenado.** Solo contempla marcación hombre a hombre con ayuda en la pintura; no modela zonas, trampas, ni cambios de marca.
4. **La acción recomendada no usa el modelo.** Se decide con geometría sobre las posiciones dibujadas. Puede haber desacuerdos visibles (por ejemplo "ACE_PROBABLE" con 59 % de probabilidad de ganar el punto, porque la etiqueta se basa solo en la distancia que debe correr el restador).
5. **Supuestos de escala.** Baloncesto: 1 unidad = 0.5 pies. Tenis: x=0 y x=100 son las líneas laterales de individuales (27 pies) e y=0 / y=100 las líneas de fondo (78 pies); si el frontend usa otra escala (por ejemplo dobles o zona detrás de la línea de fondo), las distancias en pies cambian y habría que ajustar `features.py` y reentrenar.
6. **Probabilidad de pase = completar el pase**, no que el compañero termine con un buen tiro.
7. **Tenis:** el punto de bote debe estar en la caja de saque (y entre 28.6 y 50); fuera de ella el modelo asume falta casi segura. No distingue el lado del saque (caja de la derecha o de la izquierda) ni si es primer o segundo saque.
8. **Tamaño de la validación.** Las métricas salen de 16 000 situaciones sintéticas por cabeza; no hay validación contra datos reales.
