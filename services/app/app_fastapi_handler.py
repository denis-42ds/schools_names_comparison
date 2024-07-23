"""Класс FastApiHandler, который обрабатывает запрос и возвращает наименование школы."""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from constants import MODEL_PATH, DATA_PATH

logging.basicConfig(level=logging.INFO)

class FastApiHandler:

    def __init__(self):
        """Инициализация переменных класса."""

        # типы параметров запроса для проверки
        self.df = pd.DataFrame()
        self.embeddings = np.array([])
        self.data_prepare(data_path=DATA_PATH)
        self.load_model(model_path=MODEL_PATH)
        self.get_embeddings()

    def data_prepare(self, data_path: str):
        """Загрузка данных и их подготовка.
        
        Args:
            data_path (str): Путь к данным.
         """
        try:
            schools = pd.read_csv(os.path.join(DATA_PATH, 'Школы.csv'))
            sample = pd.read_csv(os.path.join(DATA_PATH, 'Примерное написание.csv'))
            logging.info('Data loaded successfully')
        except Exception as e:
            logging.error(f"Failed to load data: {e}")

        self.df = sample.merge(schools, how='left', on='school_id')
        self.df.rename(columns={'name_x': 'name', 'name_y': 'target'}, inplace=True)
        self.df['standard'] = self.df['target'] + ': ' + self.df['region']
        self.df.drop(columns=['target', 'region'], axis=1, inplace=True)
        self.df = self.df[self.df['school_id'] != 277]

    def load_model(self, model_path: str):
        """Загрузка обученной модели.

        Args:
            model_path (str): Путь до модели.
        """
        try:
            self.model = joblib.load(model_path)
            logging.info('Model loaded successfully')
            return True
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            return False

    def get_embeddings(self):
        school_names = self.df['name'].values
        logging.info(f"Number of school names: {len(school_names)}")

        self.embeddings = self.model.encode(school_names)
        logging.info("Embeddings generated successfully")

    def school_name_search(self, input_example: str, top_k: int = 5):
        """Получение наименования школы.

        Args:
            model_params (dict): Параметры для модели.

        Returns:
            Dict: корректное наименование школы.
        """
        res = util.semantic_search(self.model.encode(input_example), self.embeddings, top_k=top_k)

        idx = [i['corpus_id'] for i in res[0]]
        score = [i['score'] for i in res[0]]
        valid_idx = [i for i in idx if i in self.df.index]  # Filter idx based on existing df index
        if not valid_idx:
            return None  # Handle no matches (optional)
        result = (self.df.loc[idx, ['school_id', 'standard']]
                  .assign(cosine_similarity=score)
                  .drop_duplicates(subset=['school_id'])
                  .iloc[:top_k].rename(columns={'standard': 'school'}))
        return result.to_dict(orient='records')

    def handle(self, params):
        """Функция для обработки запросов API.

        Args:
            params (dict): Словарь параметров запроса.

        Returns:
            dict: Словарь, содержащий результат выполнения запроса.
        """
        try:
            input_example = params['input_example']
            top_k = params['top_k']
            logging.info(f"Receiving school name for input_example: {input_example}")
            school_name = self.school_name_search(input_example, top_k)
            response = {
                        "input_example": input_example,
                        "similar_schools": school_name
                        }
            logging.info(response)
        except KeyError as e:
            logging.error(f"KeyError while handling request: {e}")
            return {"Error": "Missing key in request"}
        except Exception as e:
            logging.error(f"Error while handling request: {e}")
            return {"Error": "Problem with request"}
        else:
            return json.dumps(response)

if __name__ == "__main__":

    # создание тестового запроса
    test_params = {
        'input_example': 'авангард',
        'top_k': 5
        }

    # создание обработчика запросов для API
    handler = FastApiHandler()

    # осуществление тестового запроса
    response = handler.handle(test_params)
    print(f"Response: {response}")

