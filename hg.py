import pandas as pd
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class HierarchicalCategorySearch:
    def __init__(self, df: pd.DataFrame, create_emb_func, path_save: str):
        """
        Инициализация системы иерархического поиска категорий
        
        Args:
            df: DataFrame с колонками cat1, cat2, cat3, cat4
            create_emb_func: функция для создания эмбеддингов create_emb(texts) -> embeddings
            path_save: путь для сохранения индексов
        """
        self.df = df.copy()
        self.create_emb = create_emb_func
        self.path_save = Path(path_save)
        self.path_save.mkdir(parents=True, exist_ok=True)
        
        # Словари для хранения индексов и текстов для каждого уровня
        self.indexes = {}
        self.texts = {}
        self.hierarchy_map = {}  # для связи между уровнями
        
    def build_indexes(self):
        """Построение FAISS индексов для всех уровней категорий"""
        print("Начинаем построение индексов...")
        
        for level in range(1, 5):  # cat1, cat2, cat3, cat4
            cat_col = f'cat{level}'
            print(f"Обрабатываем уровень {cat_col}...")
            
            # Получаем уникальные категории для текущего уровня
            if level == 1:
                # Для первого уровня берем все уникальные cat1
                unique_cats = self.df[cat_col].dropna().unique()
                parent_info = None
            else:
                # Для остальных уровней создаем mapping с родительскими категориями
                parent_cols = [f'cat{i}' for i in range(1, level)]
                group_cols = parent_cols + [cat_col]
                
                # Группируем по родительским категориям и текущему уровню
                grouped = self.df[group_cols].dropna().drop_duplicates()
                unique_cats = grouped[cat_col].unique()
                
                # Создаем mapping родитель -> дети
                parent_info = {}
                for _, row in grouped.iterrows():
                    parent_key = tuple(row[parent_cols].values)
                    child = row[cat_col]
                    if parent_key not in parent_info:
                        parent_info[parent_key] = []
                    if child not in parent_info[parent_key]:
                        parent_info[parent_key].append(child)
            
            # Создаем эмбеддинги для уникальных категорий
            cat_texts = list(unique_cats)
            embeddings = self.create_emb(cat_texts)
            
            # Создаем FAISS индекс
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Косинусное сходство
            
            # Нормализуем эмбеддинги для косинусного сходства
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            # Сохраняем индекс и тексты
            self.indexes[level] = index
            self.texts[level] = cat_texts
            self.hierarchy_map[level] = parent_info
            
            # Сохраняем на диск
            index_path = self.path_save / f'index_cat{level}.faiss'
            texts_path = self.path_save / f'texts_cat{level}.pkl'
            hierarchy_path = self.path_save / f'hierarchy_cat{level}.pkl'
            
            faiss.write_index(index, str(index_path))
            with open(texts_path, 'wb') as f:
                pickle.dump(cat_texts, f)
            with open(hierarchy_path, 'wb') as f:
                pickle.dump(parent_info, f)
            
            print(f"Уровень {cat_col}: {len(cat_texts)} категорий, индекс сохранен")
        
        print("Построение индексов завершено!")
    
    def load_indexes(self):
        """Загрузка сохраненных индексов"""
        print("Загружаем индексы...")
        
        for level in range(1, 5):
            index_path = self.path_save / f'index_cat{level}.faiss'
            texts_path = self.path_save / f'texts_cat{level}.pkl'
            hierarchy_path = self.path_save / f'hierarchy_cat{level}.pkl'
            
            if index_path.exists() and texts_path.exists():
                # Загружаем индекс
                self.indexes[level] = faiss.read_index(str(index_path))
                
                # Загружаем тексты
                with open(texts_path, 'rb') as f:
                    self.texts[level] = pickle.load(f)
                
                # Загружаем иерархию
                if hierarchy_path.exists():
                    with open(hierarchy_path, 'rb') as f:
                        self.hierarchy_map[level] = pickle.load(f)
                else:
                    self.hierarchy_map[level] = None
                
                print(f"Загружен уровень cat{level}: {len(self.texts[level])} категорий")
            else:
                raise FileNotFoundError(f"Индекс для уровня cat{level} не найден")
        
        print("Загрузка индексов завершена!")
    
    def find_cascade_matches(self, text_exp: str, top_k: int = 5) -> Dict:
        """
        Каскадный поиск по всем уровням категорий
        
        Args:
            text_exp: входной текст для поиска
            top_k: количество ближайших соседей для поиска
            
        Returns:
            Словарь с результатами поиска по каждому уровню
        """
        results = {}
        current_parent = None
        
        # Создаем эмбеддинг для входного текста
        query_embedding = self.create_emb([text_exp])
        faiss.normalize_L2(query_embedding)
        
        for level in range(1, 5):
            print(f"Поиск на уровне cat{level}...")
            
            if level == 1:
                # Для первого уровня ищем среди всех cat1
                index = self.indexes[level]
                texts = self.texts[level]
                
                scores, indices = index.search(query_embedding.astype('float32'), top_k)
                
                matches = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1:  # -1 означает, что совпадение не найдено
                        matches.append({
                            'text': texts[idx],
                            'score': float(score),
                            'index': int(idx)
                        })
                
                results[level] = matches
                
                # Запоминаем лучшее совпадение как родителя для следующего уровня
                if matches:
                    current_parent = (matches[0]['text'],)
            
            else:
                # Для остальных уровней ищем только среди потомков текущего родителя
                if current_parent is None or self.hierarchy_map[level] is None:
                    results[level] = []
                    continue
                
                # Получаем список потомков для текущего родителя
                children = self.hierarchy_map[level].get(current_parent, [])
                
                if not children:
                    results[level] = []
                    continue
                
                # Находим индексы потомков в общем списке текстов
                all_texts = self.texts[level]
                child_indices = []
                child_texts = []
                
                for child in children:
                    if child in all_texts:
                        idx = all_texts.index(child)
                        child_indices.append(idx)
                        child_texts.append(child)
                
                if not child_indices:
                    results[level] = []
                    continue
                
                # Получаем эмбеддинги только для потомков
                index = self.indexes[level]
                
                # Ищем среди всех, но потом фильтруем только потомков
                scores, indices = index.search(query_embedding.astype('float32'), 
                                             min(len(all_texts), top_k * 3))
                
                matches = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx in child_indices:
                        matches.append({
                            'text': all_texts[idx],
                            'score': float(score),
                            'index': int(idx)
                        })
                        if len(matches) >= top_k:
                            break
                
                results[level] = matches[:top_k]
                
                # Обновляем родителя для следующего уровня
                if matches:
                    current_parent = current_parent + (matches[0]['text'],)
        
        return results
    
    def find_single_level(self, text_exp: str, level: int, 
                         parent_path: Optional[Tuple] = None, top_k: int = 5) -> List[Dict]:
        """
        Поиск на конкретном уровне с опциональным ограничением по родителю
        
        Args:
            text_exp: входной текст
            level: уровень категорий (1-4)
            parent_path: путь к родительской категории (например, ('cat1_value', 'cat2_value'))
            top_k: количество результатов
            
        Returns:
            Список найденных совпадений
        """
        if level not in self.indexes:
            raise ValueError(f"Индекс для уровня {level} не найден")
        
        query_embedding = self.create_emb([text_exp])
        faiss.normalize_L2(query_embedding)
        
        index = self.indexes[level]
        all_texts = self.texts[level]
        
        if level == 1 or parent_path is None:
            # Поиск среди всех категорий уровня
            scores, indices = index.search(query_embedding.astype('float32'), top_k)
            
            matches = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    matches.append({
                        'text': all_texts[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            return matches
        
        else:
            # Поиск среди потомков конкретного родителя
            hierarchy = self.hierarchy_map[level]
            if hierarchy is None:
                return []
            
            children = hierarchy.get(parent_path, [])
            if not children:
                return []
            
            # Поиск среди всех с последующей фильтрацией
            scores, indices = index.search(query_embedding.astype('float32'), 
                                         min(len(all_texts), top_k * 3))
            
            matches = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and all_texts[idx] in children:
                    matches.append({
                        'text': all_texts[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
                    if len(matches) >= top_k:
                        break
            
            return matches[:top_k]
    
    def get_full_path(self, cat4_text: str) -> Optional[Tuple]:
        """
        Получить полный путь категории по значению cat4
        
        Args:
            cat4_text: текст категории 4 уровня
            
        Returns:
            Кортеж (cat1, cat2, cat3, cat4) или None если не найдено
        """
        row = self.df[self.df['cat4'] == cat4_text]
        if len(row) > 0:
            return tuple(row.iloc[0][['cat1', 'cat2', 'cat3', 'cat4']].values)
        return None

# Пример использования:
"""
# Инициализация
searcher = HierarchicalCategorySearch(df, create_emb, "path/to/save")

# Построение индексов (выполняется один раз)
searcher.build_indexes()

# Или загрузка существующих индексов
searcher.load_indexes()

# Каскадный поиск
results = searcher.find_cascade_matches("ваш поисковый текст")

# Результат будет содержать найденные категории для каждого уровня:
for level, matches in results.items():
    print(f"Уровень cat{level}:")
    for match in matches:
        print(f"  {match['text']} (score: {match['score']:.3f})")

# Поиск на конкретном уровне
level2_results = searcher.find_single_level("текст", level=2, parent_path=("найденная_cat1",))
"""
