
# Anchor Optical TrackeR

## Overview
This project implements an optical flow tracking system for anchors using OpenCV. It estimates positional offsets and angles through keypoint tracking and applies Kalman filtering for noise reduction.

## Features
- Real-time tracking of visual features using Lucas-Kanade optical flow.
- Kalman filter to smooth positional and angular estimates.
- Toggle tracking on/off using a flag file.
- Saves filtered offsets and angle data in JSON format.
- Logs important events and errors.
- Implements waypoint navigation and updates.

## Rope (Веревочный) Method Description
The rope method simulates a physical rope connecting the anchor point to a target position. The system calculates tension and angular displacement based on the optical flow offset data, providing a more intuitive and realistic approach to positional tracking in environments where tethered movement is relevant.

## Usage
- Run the main script `python main.py`.
- Use keyboard controls:
  - `t`: toggle tracking on/off.
  - `r`: reinitialize tracking points.
  - `q`: quit the application.

## Requirements
- Python 3.x
- OpenCV
- Numpy

## License
MIT License

---

# Anchor Optical TrackeR

## Обзор
Проект реализует систему трекинга оптического потока для якорей с использованием OpenCV. Оценивает смещения и углы положения через отслеживание ключевых точек и применяет фильтр Калмана для снижения шума.

## Функции
- Реальное время трекинга визуальных признаков с помощью оптического потока Лукаса-Канаде.
- Фильтр Калмана для сглаживания оценки позиции и угла.
- Включение/выключение трекинга через файл-флаг.
- Сохранение фильтрованных данных о смещении и угле в формате JSON.
- Логирование важных событий и ошибок.
- Навигация с использованием waypoint'ов и их обновление.

## Описание веревочного метода
Веревочный метод моделирует физическую веревку, соединяющую точку якоря с целевой позицией. Система рассчитывает натяжение и угловое смещение на основе данных оптического потока, обеспечивая более интуитивный и реалистичный подход к позиционному трекингу в средах, где важно движение с привязкой.

## Использование
- Запуск основного скрипта `python main.py`.
- Управление с клавиатуры:
  - `t`: включение/выключение трекинга.
  - `r`: повторная инициализация точек трекинга.
  - `q`: выход из приложения.

## Требования
- Python 3.x
- OpenCV
- Numpy

## Лицензия
MIT License
