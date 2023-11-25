import datetime
from jose import jwt
import aiohttp

from app.config.settings import get_settings

settings = get_settings()


def generate_token():
    expires = datetime.datetime.now() + datetime.timedelta(days=3600)
    expires_timestamp = expires.timestamp()
    to_encode = {"exp": expires_timestamp}
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


token_predict = generate_token()


async def send_post_request(data, port):
    url = f"{settings.api_models_base}:{port}/models/predict"
    timeout = aiohttp.ClientTimeout(total=7200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
                url,
                json=data,
                headers={'Authorization': f'Bearer {token_predict}'}

        ) as response:
            return await response.json()
