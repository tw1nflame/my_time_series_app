import os
from pathlib import Path
from dotenv import set_key, dotenv_values
from .settings import settings, reload_env_vars

def validate_secret_key(key: str) -> bool:
    """
    Проверяет, совпадает ли переданный ключ с SECRET_KEY из настроек.
    
    Args:
        key: Ключ для проверки
        
    Returns:
        bool: True, если ключ совпадает, иначе False
    """
    return key == settings.SECRET_KEY


async def update_env_variables(env_vars: dict) -> bool:
    """
    Обновляет переменные окружения в файле .env
    
    Args:
        env_vars: Словарь с переменными для обновления
        
    Returns:
        bool: True, если обновление успешно, иначе False
    """
    try:
        # Путь к файлу .env
        env_path = Path(__file__).parent.parent / '.env'
        
        # Проверяем, что файл существует
        if not env_path.exists():
            return False
        
        # Проверяем, что все необходимые ключи присутствуют
        required_keys = ['DB_USER', 'DB_PASS', 'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_SCHEMA']
        if not all(key in env_vars for key in required_keys):
            return False
              # Обновляем каждую переменную в файле .env
        for key, value in env_vars.items():
            if key in required_keys:
                set_key(str(env_path), key, value)
        
        # Перезагружаем переменные окружения после обновления файла .env
        reload_env_vars()
        
        # Обновляем объект настроек, чтобы он использовал новые значения
        settings.refresh()
                
        return True
    except Exception:
        return False