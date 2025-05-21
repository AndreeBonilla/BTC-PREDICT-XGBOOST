# Predicción del Precio de Bitcoin con XGBoost

Este proyecto utiliza modelos de aprendizaje automático (ML), específicamente el algoritmo **XGBoost**, para predecir el precio futuro del **Bitcoin (BTC)**. Se entrenó el modelo con datos históricos desde 2023 hasta 2025 y se generó una predicción para los próximos 30 días.

## Tecnologías y Librerías Utilizadas

- Python
- yfinance (para descarga de datos financieros)
- pandas y numpy
- scikit-learn
- XGBoost
- matplotlib (para visualización de resultados)

## Resultados del Modelo

- **MSE (Error Cuadrático Medio):** 641,460.05  
- **MAE (Error Absoluto Medio):** 397.39  
- **Error Porcentual Medio:** 0.43%

El modelo mostró un buen desempeño con un bajo margen de error en comparación con los valores reales, generando predicciones ajustadas para los siguientes días.

## Metodología

1. **Descarga de datos** históricos de BTC usando `yfinance`.
2. **Procesamiento y normalización** de los datos.
3. **Entrenamiento del modelo** XGBoost con una ventana de entrada de 2 días anteriores para predecir el siguiente.
4. **Predicción futura** de los siguientes 30 días.
5. **Visualización de resultados** en un gráfico comparativo.

## Gráfico de Predicción

El gráfico generado muestra una curva de precios proyectados para BTC, comparada con los valores reales más recientes, permitiendo analizar tendencias y posibles variaciones a corto plazo.

## Próximos Pasos

- Incorporar más variables (técnicas o fundamentales) en el modelo.
- Probar otros algoritmos (LSTM, Random Forest, Prophet).
- Hacer una versión web interactiva del modelo.
- Adaptarlo para otras criptomonedas (ETH, ADA, SOL...).

## Autor

**Jhonny Andres Bonilla Cobacango**  
Estudiante de Matemáticas Aplicadas  
Apasionado por la inteligencia artificial, finanzas y predicción de datos financieros.  
🔗 [LinkedIn](https://www.linkedin.com/in/andresbonilla442001/)  
