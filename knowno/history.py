import redis 
import json 
import uuid 

class ChatHistory:
     def __init__(self, host: str, port: int, db: int):
          self.redis = redis.Redis(host=host, port=port, db=db)
          self.current_session_id = None
     
     def start_session(self):
          self.current_session_id = str(uuid.uuid4())[:8]
          key = f'session:{self.current_session_id}'

          initial_data = {
               'session_id': self.current_session_id,
               'history': [],
          }

          self.redis.set(key, json.dump(initial_data), ex=3600)
          return initial_data 

     def add_message(self, user_message, robot_message):
          if not self.current_session_id:
               return False 
          
          key = f'session:{self.current_session_id}'
          data = json.loads(self.redis.get(key))

          chat_turn = {
               'session_id': self.current_session_id,
               'user': user_message,
               'robot_kitchen': robot_message,
          }

          data['history'].append(chat_turn)
          self.redis.set(key, json.dumps(data), ex=3600)
          return True 
     
     def end_session(self):
          if self.current_session_id:
               self.redis.delete(f'session:{self.current_session_id}')
               temp_id = self.current_session_id
               self.current_session_id = None
               return temp_id
          return None 

     