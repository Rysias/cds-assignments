from pathlib import Path
import src.util as util
import openai


def get_api_key(config: Path, keyname: str = "goose_api") -> str:
    """
    Gets the API key from the config file.
    """
    return util.read_json(config)[keyname]


def authenticate_goose(config: Path) -> None:
    """
    Authenticates with the goose API.
    """
    api_key = get_api_key(config, keyname="goose_api")
    openai.api_key = api_key
    openai.api_base = "https://api.goose.ai/v1"
