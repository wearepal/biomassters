from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

__all__ = ["StaticConfSpPlugin"]


class StaticConfSpPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(provider="bio", path="file://static_conf")
