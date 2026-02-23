import json
import re
import sys
from pathlib import Path

# Ensure project root is on path when running this script directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from llm import LLM
from knowno.embedding import EnvironmentMatcher, EmbeddingSelector
from knowno.classify import AmbiguityClassifier

class EntityExtractor:    
    def __init__(self, llm_instance):
        """
        :param llm_instance: Instance của lớp LLM (Ollama) đã thiết lập trước đó.
        """
        self.llm = llm_instance

    def extract(self, query):
        """
        Trích xuất các vật thể (objects) từ câu lệnh.
        
        :param query: Câu lệnh tự nhiên (ví dụ: "Lấy sữa trong tủ lạnh rót ra cốc sứ")
        :return: List các objects (ví dụ: ["sữa", "tủ lạnh", "cốc sứ"])
        """
        prompt = f"""
        [CONTEXT]
        You are a semantic parser for a kitchen robot. 
        Extract ONLY the physical objects / nouns (kitchen items, tools, food) mentioned in the query.

        [TASK]
        Query: "{query}"

        [OUTPUT FORMAT]
        Return ONLY a JSON array of strings - the list of objects found.
        Example: ["milk", "glass mug"] or ["clean sponge", "dish soap", "plate"]

        JSON:
        """
        
        response_text = self.llm.generate(prompt)
        
        try:
            # Ưu tiên tìm JSON array [...]
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                objects = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text)
                objects = parsed.get("objects", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(objects, list):
                objects = [objects] if objects else []
            return [str(o).lower().strip() for o in objects]
        except Exception as e:
            print(f"[Error] Không thể parse JSON từ LLM: {e}")
            return []


class TaskHandler:
    """
    Xử lý task theo hai giai đoạn:
    - task: nhập 1 lần duy nhất, chỉ lưu, không extract.
    - query: mỗi bước (handle_step) mới gọi extract.
    """

    def __init__(self, extractor, env_matcher, embedding_selector, classifier=None):
        self.extractor = extractor
        self.env_matcher = env_matcher
        self.embedding_selector = embedding_selector
        self.classifier = classifier
        self.current_task = None  # Chỉ set 1 lần qua start_task(task)
        self.environment = None   # Được set trong start_task dựa trên task
        self.entities = None      # Được set mỗi lần handle_step(query)

    def start_task(self, task):
        """
        Giai đoạn 1: Tiếp nhận task tổng quát (chỉ gọi 1 lần).
        Không thực hiện extract, chỉ lưu task và tìm environment tương ứng.
        """
        self.current_task = task
        # Tìm environment dựa trên task
        self.environment = self.env_matcher.find_environment(task)
        
        return (
            f"Tôi đã nhận nhiệm vụ. Bây giờ, hãy cung cấp các bước cụ thể bạn muốn tôi thực hiện." )

    def handle_step(self, step_query, environment=None):
        """
        Giai đoạn 2: Xử lý từng bước (query). Với query thì thực hiện extract.
        Tìm top 5 vật thể phù hợp nhất từ environment và classify ambiguity.
        """
        # Extract entities từ step query
        self.entities = self.extractor.extract(step_query)
        
        # Sử dụng environment từ start_task hoặc tham số truyền vào
        env = environment if environment is not None else self.environment
        
        if not env:
            print("[Warning] Không có environment để tìm vật thể")
            return {
                "entities": self.entities,
                "top_objects": [],
                "classification": None
            }
        
        top_objects = self.embedding_selector.select_top_objects(
            self.entities, 
            env, 
            top_k=3 
        )
        
        # Classify ambiguity nếu có classifier
        classification = None
        if self.classifier:
            classification = self.classifier.classify(
                self.current_task,
                step_query,
                self.entities,
                top_objects
            )
        
        return {
            "entities": self.entities,
            "top_objects": top_objects,
            "classification": classification
        }


if __name__ == "__main__":
    # Model generate: dùng cho extract entities và classify ambiguity
    dataset_path = Path(__file__).resolve().parent.parent / "ambik_dataset" / "ambik_test_900.csv"

    llm = LLM("ollama:llama3.1:8b", {"max_new_tokens": 150, "temperature": 0.7})
    embed_model = LLM("ollama:mxbai-embed-large:latest", {})
    
    # Trích xuất entity từ step task current và tìm environment phù hợp của task đó
    extractor = EntityExtractor(llm)
    env_matcher = EnvironmentMatcher(dataset_path)
    
    # Thực hiện tìm top K vật thể phù hợp dựa vào task current và thông tin từ môi trường
    embedding_selector = EmbeddingSelector(embed_model)
    classifier = AmbiguityClassifier(llm)
    
    handler = TaskHandler(extractor, env_matcher, embedding_selector, classifier)

    task = input("Enter task: ")
    print(handler.start_task(task))

    while True:
        query = input("\nEnter step query (hoặc 'quit' để kết thúc): ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            break
        
        result = handler.handle_step(query)
        
        print(f"\nObjects extracted: {result['entities']}")
        
        # In top objects với score
        print(f"Top objects:")
        for i, (obj, score) in enumerate(result['top_objects'], 1):
            print(f"  {i}. {obj:<30} (score: {score:.4f})")
        
        # Hiển thị classification nếu có
        if result.get('classification'):
            cls = result['classification']
            print(f"\nClassification: {cls['status']}")
            print(f"  Label: {cls['label']}")