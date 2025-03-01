# Análisis sobre la Importancia de Mantener los Registros Históricos de los Clientes en el Modelo de ML

## Introducción
Durante el desarrollo de un modelo de Machine Learning para predecir el puntaje crediticio de los clientes, surgió una interrogante fundamental: ¿es correcto dejar un solo registro por cliente? En un principio, se realizó una limpieza exhaustiva de los datos, eliminando duplicados y dejando un único registro por cliente, lo que redujo significativamente el dataset de 100,000 registros a solo 7,500 clientes. Sin embargo, al evaluar el desempeño del modelo, se obtuvo una precisión del 99%, lo que generó sospechas de sobreajuste.

## La Importancia de los Registros Históricos
Al analizar el problema en mayor profundidad, se comprendió que eliminar registros históricos de un mismo cliente era contraproducente. Cada cliente posee múltiples registros que representan su comportamiento financiero a lo largo del tiempo, y este historial es clave para predecir su puntaje crediticio. Si se elimina esta información, se pierde la oportunidad de identificar tendencias y patrones de comportamiento.

Por ejemplo, si un cliente tiene múltiples registros mensuales con distintas variables financieras, estos datos ayudan a entender su estabilidad económica y su evolución. La agregación de un solo registro por cliente omite esta variabilidad y puede llevar a la pérdida de información crucial para el modelo.

## Evaluación del Modelo y Resultados Sospechosos
Después de decidir mantener múltiples registros por cliente y reentrenar el modelo, los resultados obtenidos fueron extremadamente altos en precisión:

| Modelo | Accuracy | AUC | Recall | Precisión | F1 Score | Kappa | MCC |
|--------|----------|------|--------|------------|----------|-------|-----|
| Decision Tree Classifier | 99.98% | 99.98% | 99.98% | 99.98% | 99.98% | 99.97% | 99.97% |
| Random Forest Classifier | 99.98% | 100.00% | 99.98% | 99.98% | 99.98% | 99.97% | 99.97% |
| CatBoost Classifier | 99.98% | 100.00% | 99.98% | 99.98% | 99.98% | 99.97% | 99.97% |
| LightGBM | 99.96% | 100.00% | 99.96% | 99.96% | 99.96% | 99.94% | 99.94% |
| Extra Trees Classifier | 99.34% | 99.99% | 99.34% | 99.35% | 99.34% | 98.93% | 98.93% |

Estos resultados, aunque impresionantes, levantan señales de alerta. Un modelo de Machine Learning rara vez alcanza una precisión perfecta en datos reales, lo que sugiere que el modelo podría estar memorizando patrones en lugar de aprender reglas generales.

## Preguntas Clave y Escenarios Sospechosos
Ante estos resultados, se plantean las siguientes preguntas para evaluar la validez del modelo:

1. **¿Existe una fuga de datos?**
   - Si alguna variable en el dataset contiene indirectamente el puntaje de crédito, el modelo simplemente lo memorizó en lugar de aprender patrones generales.

2. **¿Cómo se desempeña el modelo en datos completamente nuevos?**
   - Se creó un dataset especial llamado *El_gato_de_Schrödinger.csv*, que contiene 1,000 clientes que nunca fueron vistos por el modelo durante el entrenamiento.
   - Si el modelo sigue teniendo un 99% de precisión en estos datos, sin ninguna fuga, significaría que es increíblemente bueno. Sin embargo, esto es poco probable en un problema de clasificación real.

3. **¿Qué métricas adicionales se pueden analizar?**
   - La matriz de confusión puede revelar si el modelo está sesgado hacia una clase específica.
   - La importancia de las características puede ayudar a identificar si hay alguna variable dominante que explique la alta precisión.

4. **¿Cómo se desempeñan otros modelos más simples?**
   - Se probó una regresión logística y obtuvo una precisión de apenas 49.59%, lo que indica que el problema podría ser más complejo de lo que parece.

## Conclusión
El análisis sugiere que la eliminación de registros históricos de los clientes no es recomendable, ya que estos registros reflejan el comportamiento financiero a lo largo del tiempo y son esenciales para una predicción precisa. Además, los resultados del modelo levantan sospechas de sobreajuste o fuga de datos, lo que debe ser investigado antes de su implementación en un entorno real.

La validación en datos completamente nuevos, la revisión de la matriz de confusión y el análisis de la importancia de las características serán pasos cruciales para garantizar que el modelo sea verdaderamente predictivo y no simplemente una máquina de memorización.

