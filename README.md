# Название проекта: Создание решения по стандартизации названий спортивных школ.

## Статус проекта: в работе

## Заказчик: Компания GoProtect

## Описание рабочих файлов и директорий:
- [school_names.ipynb](https://github.com/denis-42ds/schools_names_comparison/blob/development/school_names.ipynb) - рабочая тетрадь с исследованиями, визуализациями и текстовыми описаниями
- [requirements.txt](https://github.com/denis-42ds/schools_names_comparison/blob/development/requirements.txt) - список зависимостей, необходимых для работы проекта, а также их версии
- [research_functions.py](https://github.com/denis-42ds/schools_names_comparison/blob/development/research_functions.py) - скрипт с функциями для проведения исследования
- [assets](https://github.com/denis-42ds/schools_names_comparison/tree/development/assets) - директория с сохранёнными артефактами
- [services](https://github.com/denis-42ds/schools_names_comparison/tree/development/services) - директория с приложением

## Установка зависимостей и просмотр исследования
```Bash
git clone https://github.com/denis-42ds/schools_names_comparison.git
cd schools_names_comparison
pip install -r requirements.txt
jupyter lab
```

## Запуск FastAPI-микросервиса

```
git clone https://github.com/denis-42ds/schools_names_comparison.git
cd schools_names_comparison/services
docker compose up --build
```

Для просмотра документации API и совершения тестовых запросов пройти по ссылке: [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs)
<br>Доступ к экспозиции метрик `Prometheus`: [http://localhost:8081/metrics](http://localhost:8081/metrics)
<br>Доступ к веб-интерфейсу `Prometheus`: [http://localhost:9090](http://localhost:9090)
<br>Доступ к веб-интерфейсу `Grafana`: [http://localhost:3000](http://localhost:3000)
<br>Для остановки приложения: ```docker compose down``` или `Press CTRL+C to quit`

[Демонстрация работы приложения]()

## Описание проекта

Сервис "Мой Чемпион" помогает спортивным школам фигурного катания, тренерам
мониторить результаты своих подопечных и планировать дальнейшее развитие спортсменов.

## Цель

- Создать решение для стандартизации названий спортивных школ.
  <br>Например, одна и та же школа может быть записана по-разному
  <br>Необхдодимо сопоставить эти варианты эталонному названию из предоставленной таблицы

## Задачи

- Изучить данные – эталонные названия СШ и варианты пользовательского ввода
- Подготовить обучающий набор данных на основе эталонного датасета
- Создать модель для подбора наиболее вероятных названий при ошибочном вводе
- Создать функцию (класс, модуль) для применения в сервисе
  - возможность выбора количества кандидатов
  - вывод в виде списка словарей
- Протестировать решение
- Проанализировать результат и предложить варианты улучшения
- Создать документацию
  - описание признаков
  - какая модель используется
  - как оценивается качество
  - инструкция по запуску (применению)
- Создать демо приложение

## Используемые инструменты
- python: ;
- mlflow;
- postgresql;
- bash;
- fastapi, grafana, prometheus

## Заключение:

Итоги проделанного исследования.


