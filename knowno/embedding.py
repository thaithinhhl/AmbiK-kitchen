import csv
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class EnvironmentMatcher:
    """
    Tìm environment phù hợp với task từ dataset.
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.data = self.load_data()
    
    def load_data(self):
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def find_environment(self, task):
        best_match = None
        max_overlap = 0
        
        task_lower = task.lower()
        
        for row in self.data:
            # Kiểm tra các cột task có thể
            candidates = [
                row.get('ambiguous_task', ''),
                row.get('unambiguous_direct', ''),
                row.get('unambiguous_indirect', '')
            ]
            
            for candidate in candidates:
                if not candidate:
                    continue
                    
                candidate_lower = candidate.lower()
                
                # Đếm số từ chung
                task_words = set(task_lower.split())
                candidate_words = set(candidate_lower.split())
                overlap = len(task_words & candidate_words)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = row
        
        if best_match:
            # Lấy environment_short và parse thành list
            env_str = best_match.get('environment_short', '')
            # Parse "bread knife, paring knife, ..." thành list
            environment = [item.strip() for item in env_str.split(',')]
            return environment
        else:
            print(f"[Warning] Không tìm thấy environment phù hợp với task: {task}")
            return []


class EmbeddingSelector:
    """
    Embedding và tính cosine similarity để chọn top K vật thể phù hợp nhất.
    """
    
    def __init__(self, llm_instance):
        self.llm = llm_instance
    
    def get_embedding(self, text):
        try:
            embedding = self.llm.embed(text)
            return np.array(embedding)
        except Exception as e:
            print(f"[Error] Không thể lấy embedding cho text '{text}': {e}")
            # Fallback: trả về vector zero
            return np.zeros(384)
    
    def select_top_objects(self, extracted_objects, environment_objects, top_k=3):
        """
        Lấy Top-K cho từng vật thể rồi gộp lại.
        
        :param extracted_objects: List các objects đã extract từ step query
        :param environment_objects: List các vật thể trong environment
        :param top_k: Số lượng vật thể cần chọn cho mỗi extracted object
        :return: List [(object, score), ...] - gộp tất cả top-K, giới hạn 10 vật thể
        """
        if not environment_objects or not extracted_objects:
            return []

        # Embedding tất cả environment objects một lần
        env_embeddings = np.array([self.get_embedding(obj) for obj in environment_objects])
        all_candidates = {}

        # Lấy top-K cho từng extracted object
        for obj in extracted_objects:
            obj_emb = self.get_embedding(obj).reshape(1, -1)
            similarities = cosine_similarity(obj_emb, env_embeddings)[0]
            
            # Lấy top_k cho riêng vật thể này
            top_indices = np.argsort(similarities)[::-1][:top_k]
            for idx in top_indices:
                name = environment_objects[idx]
                score = float(similarities[idx])
                # Giữ lại score cao nhất nếu vật thể xuất hiện nhiều lần
                if name not in all_candidates or score > all_candidates[name]:
                    all_candidates[name] = score

        # Loại bỏ các object trùng với extracted_objects
        extracted_lower = {o.lower().strip() for o in extracted_objects}
        filtered_candidates = {
            name: score
            for name, score in all_candidates.items()
            if name.lower().strip() not in extracted_lower
        }

        # Nếu lọc hết thì dùng lại all_candidates
        final_candidates = filtered_candidates or all_candidates

        # Trả về danh sách đã gộp và sắp xếp theo score
        sorted_candidates = sorted(final_candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:10]
