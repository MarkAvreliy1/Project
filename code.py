import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class FashionAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.filename)

            self.df = self.df.fillna(0).drop_duplicates()
            logging.info("Данные загружены и очищены.")
            return True
        except Exception as e:
            logging.error(f"Ошибка загрузки: {e}")
            return False

    def print_text_stats(self):
        print("\n" + "="*40)
        print(" ОСНОВНАЯ СТАТИСТИКА ")
        print("="*40)
        print(self.df.describe())
        
        if 'Popularity_Score' in self.df.columns:
            print("\nМедиана популярности по категориям:")
            popularity = self.df.groupby('Category')['Popularity_Score'].median().sort_values(ascending=False)
            print(popularity)

    def plot_all_charts(self):

        sns.set_theme(style="whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Комплексный анализ зимних модных трендов (2023-2025)', fontsize=20)

        sns.countplot(ax=axes[0, 0], data=self.df, x='Color', 
                      order=self.df['Color'].value_counts().index, palette='muted', hue='Color', legend=False)
        axes[0, 0].set_title('Самые трендовые цвета')
        axes[0, 0].tick_params(axis='x', rotation=45)


        brand_price = self.df.groupby('Brand')['Price(USD)'].mean().sort_values()
        brand_price.plot(kind='barh', ax=axes[0, 1], color='skyblue')
        axes[0, 1].set_title('Средняя цена по брендам (USD)')

        sns.boxplot(ax=axes[1, 0], data=self.df, x='Style', y='Popularity_Score', palette='Set2', hue='Style', legend=False)
        axes[1, 0].set_title('Разброс популярности в зависимости от стиля')


        sns.lineplot(ax=axes[1, 1], data=self.df, x='Season', y='Customer_Rating', marker='o', color='red')
        axes[1, 1].set_title('Динамика рейтинга покупателей по сезонам')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def run(self):
        if self.load_data():
            self.print_text_stats()
            self.plot_all_charts()
            logging.info("Анализ завершен.")

if __name__ == "__main__":

    PATH = 'Winter_Fashion_Trends_Dataset.csv'
    worker = FashionAnalyzer(PATH)
    worker.run()
