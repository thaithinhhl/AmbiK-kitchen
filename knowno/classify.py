import json
import re


class AmbiguityClassifier:
    """
    Phân loại step query là Unambiguous hoặc Ambiguous (Preferences, Common Sense, Safety)
    dựa trên task, entities, và top-K objects.
    """
    
    def __init__(self, llm_instance, prompt_path=None):
        """
        :param llm_instance: Instance của LLM
        :param prompt_path: Đường dẫn đến file prompt template (mặc định: prompts/choising.txt)
        """
        self.llm = llm_instance
        self.prompt_template = self.load_prompt(prompt_path)
    
    def load_prompt(self, prompt_path=None):
        """Đọc prompt template từ file."""
        if prompt_path is None:
            from pathlib import Path
            prompt_path = Path(__file__).parent / "prompts" / "choising.txt"
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[Warning] Không thể đọc prompt từ {prompt_path}: {e}")
            return None
    
    def classify(self, task, step_query, entities, top_objects):
        """
        Phân loại step query.
        
        :param task: Task tổng quát (string)
        :param step_query: Step query hiện tại (string)
        :param entities: Dict với keys 'actions' và 'objects'
        :param top_objects: List các objects phù hợp [(obj, score), ...] hoặc [obj, ...]
        :return: Dict với keys: status, label
        """
        # Format top objects
        if top_objects and isinstance(top_objects[0], tuple):
            top_k_list = [obj for obj, _ in top_objects]
        else:
            top_k_list = list(top_objects)
        
        # Tạo prompt
        prompt = self.prompt_template.replace("<TASK>", task)
        prompt = prompt.replace("<STEP_QUERY>", step_query)
        prompt = prompt.replace("<ENTITIES>", json.dumps(entities))
        prompt = prompt.replace("<TOP_K_OBJECTS>", json.dumps(top_k_list))
        
        # Gọi LLM
        response_text = self.llm.generate(prompt)
        
        # Parse JSON 
        try:
            # Tìm tất cả các JSON objects trong response
            json_matches = list(re.finditer(r'\{[^{}]*\}', response_text))
            
            if json_matches:
                # Lấy JSON object cuối cùng (thường là kết quả cuối)
                json_str = json_matches[-1].group()
                result = json.loads(json_str)
            else:
                # Fallback: thử parse toàn bộ response
                result = json.loads(response_text)
            
            # Validate và normalize
            status = result.get("status", "Unambiguous")
            label = result.get("label", "None")
            
            return {
                "status": status,
                "label": label,
            }
            
        except Exception as e:
            print(f"[Error] Không thể parse JSON từ LLM: {e}")
            print(f"[Debug] Response: {response_text}")
            return {
                "status": "Unambiguous",
                "label": "None",
            }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    
    from llm import LLM
    
    # Test
    model = LLM("ollama:llama3.1:8b", {"max_new_tokens": 150, "temperature": 0.7})
    classifier = AmbiguityClassifier(model)
    