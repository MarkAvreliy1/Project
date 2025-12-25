# Импорт необходимых библиотек
import pandas as pd          # Для работы с табличными данными
import numpy as np           # Для математических операций
import matplotlib.pyplot as plt  # Для построения графиков
import seaborn as sns        # Для статистической визуализации
import logging               # Для логирования работы программы

# Настройка логирования: вывод информационных сообщений в консоль
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class FashionAnalyzer:
    """
    Класс для анализа данных о зимних модных трендах.
    Выполняет загрузку, очистку, анализ и визуализацию
    """
    
    def __init__(self, filename):
        """
        Инициализация анализатора.
        
        Аргументы:
            filename (str): Путь к CSV с данными
        """
        self.filename = filename  # Сохраняем имя файла
        self.df = None           # Инициализируем DataFrame как None
    
    def load_data(self):
        """
        Загрузка и очистка данных из CSV-файла.
        
        Возвращает:
            bool: True если данные успешно загружены, False в случае ошибки
        """
        try:
            # Загружаем данные из CSV-файла
            self.df = pd.read_csv(self.filename)
            
            # Очищаем данные:
            # 1. Заменяем пропущенные значения на 0
            # 2. Удаляем дублирующиеся строки
            self.df = self.df.fillna(0).drop_duplicates()
            
            # Логируем успешную загрузку
            logging.info("Данные загружены и очищены.")
            return True
            
        except FileNotFoundError:
            # Обработка ошибки: файл не найден
            logging.error(f"Файл {self.filename} не найден.")
            return False
        except Exception as e:
            # Обработка любых других ошибок
            logging.error(f"Ошибка загрузки данных: {e}")
            return False
    
    def print_text_stats(self):
        """
        Вывод текстовой статистики по данным.
        Показывает общую статистику и медиану популярности по категориям.
        """
        print("\n" + "="*40)
        print(" ОСНОВНАЯ СТАТИСТИКА ")
        print("="*40)
        
        # Выводим описательную статистику для числовых колонок
        print(self.df.describe())
        
        # Если в данных есть колонка 'Popularity_Score',
        # вычисляем и выводим медиану популярности по категориям
        if 'Popularity_Score' in self.df.columns:
            print("\n" + "-"*40)
            print("Медиана популярности по категориям:")
            print("-"*40)
            
            # Группируем по категориям и считаем медиану популярности
            popularity = self.df.groupby('Category')['Popularity_Score'] \
                              .median() \
                              .sort_values(ascending=False)
            print(popularity)
    
    def plot_all_charts(self):
        """
        Построение 4 визуализаций для анализа данных.
        Создает единое окно с 4 графиками.
        """
        # Устанавливаем тему для графиков (белый фон с сеткой)
        sns.set_theme(style="whitegrid")
        
        # Создаем фигуру с 4 подграфиками (2 строки, 2 столбца)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Добавляем общий заголовок для всех графиков
        fig.suptitle('Комплексный анализ зимних модных трендов (2023-2025)', 
                    fontsize=20, fontweight='bold')
        
        #ГРАФИК 1: Самые трендовые цвета
        try:
            # Строим столбчатую диаграмму для колонки 'Color'
            sns.countplot(
                ax=axes[0, 0],        # Размещаем на первом графике
                data=self.df,          # Используем наш DataFrame
                x='Color',             # Данные для оси X
                order=self.df['Color'].value_counts().index,  # Сортируем по частоте
                palette='muted',       # Используем приглушенную палитру
                hue='Color',           # Группируем по цвету
                legend=False           # Скрываем легенду
            )
            axes[0, 0].set_title('Самые трендовые цвета', fontsize=14, fontweight='bold')
            axes[0, 0].tick_params(axis='x', rotation=45)  # Поворачиваем подписи оси X
            axes[0, 0].set_xlabel('Цвет', fontsize=12)
            axes[0, 0].set_ylabel('Количество', fontsize=12)
        except Exception as e:
            logging.warning(f"Не удалось построить график цветов: {e}")
        
        # ГРАФИК 2: Средняя цена по брендам
        try:
            # Группируем данные по брендам и вычисляем среднюю цену
            brand_price = self.df.groupby('Brand')['Price(USD)'].mean().sort_values()
            
            # Строим горизонтальную столбчатую диаграмму
            brand_price.plot(
                kind='barh',          # Горизонтальные столбцы
                ax=axes[0, 1],        # Размещаем на втором графике
                color='skyblue',      # Задаем цвет
                edgecolor='darkblue'  # Цвет границ столбцов
            )
            axes[0, 1].set_title('Средняя цена по брендам (USD)', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Средняя цена, USD', fontsize=12)
            axes[0, 1].set_ylabel('Бренд', fontsize=12)
        except Exception as e:
            logging.warning(f"Не удалось построить график цен: {e}")
        
        # ГРАФИК 3: Разброс популярности по стилям
        try:
            # Строим boxplot (ящик с усами) для анализа распределения
            sns.boxplot(
                ax=axes[1, 0],        # Размещаем на третьем графике
                data=self.df,
                x='Style',            # Категории по оси X
                y='Popularity_Score', # Значения по оси Y
                palette='Set2',       # Используем палитру Set2
                hue='Style',          # Группируем по стилю
                legend=False          # Скрываем легенду
            )
            axes[1, 0].set_title('Разброс популярности в зависимости от стиля', 
                               fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Стиль', fontsize=12)
            axes[1, 0].set_ylabel('Оценка популярности', fontsize=12)
        except Exception as e:
            logging.warning(f"Не удалось построить boxplot популярности: {e}")
        
        # ===== ГРАФИК 4: Динамика рейтинга по сезонам =====
        try:
            # Строим линейный график для отслеживания изменений
            sns.lineplot(
                ax=axes[1, 1],        # Размещаем на четвертом графике
                data=self.df,
                x='Season',           # Ось X: сезоны
                y='Customer_Rating',  # Ось Y: рейтинг покупателей
                marker='o',           # Добавляем маркеры в точках данных
                color='red',          # Цвет линии
                linewidth=2.5,        # Толщина линии
                markersize=8          # Размер маркеров
            )
            axes[1, 1].set_title('Динамика рейтинга покупателей по сезонам', 
                               fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Сезон', fontsize=12)
            axes[1, 1].set_ylabel('Средний рейтинг', fontsize=12)
            
            # Добавляем сетку для лучшей читаемости
            axes[1, 1].grid(True, alpha=0.3)
        except Exception as e:
            logging.warning(f"Не удалось построить график рейтинга: {e}")
        
        # Автоматически подгоняем расположение графиков
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Показываем все графики
        plt.show()
    
    def run(self):
        """
        Основной метод для запуска полного анализа.
        Выполняет все шаги последовательно.
        """
        # 1. Загружаем данные
        if self.load_data():
            # 2. Выводим статистику
            self.print_text_stats()
            
            # 3. Строим графики
            self.plot_all_charts()
            
            # 4. Сообщаем об успешном завершении
            logging.info("Анализ завершен успешно!")
        else:
            logging.error("Анализ не может быть выполнен из-за ошибки загрузки данных.")


if __name__ == "__main__":
    """
    Точка входа в программу.
    Запускается только при прямом выполнении этого файла.
    """
    # Указываем путь к файлу с данными
    PATH = 'Winter_Fashion_Trends_Dataset.csv'
    
    # Создаем экземпляр анализатора
    analyzer = FashionAnalyzer(PATH)
    
    # Запускаем анализ
    analyzer.run()
