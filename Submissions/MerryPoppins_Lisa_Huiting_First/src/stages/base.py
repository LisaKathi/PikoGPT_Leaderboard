from abc import ABC, abstractmethod


class Stage(ABC):
    @abstractmethod
    def run(self, **kwargs):
        ...


class StageRegistry:
    _registry: dict[str, tuple[type[Stage], str]] = {}

    @classmethod
    def register(cls, name: str, description: str):
        def decorator(stage_cls: type[Stage]) -> type[Stage]:
            cls._registry[name] = (stage_cls, description)
            return stage_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[Stage]:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown stage: {name!r}. Available: {cls.all_names()}"
            )
        return cls._registry[name][0]

    @classmethod
    def all_names(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def all_descriptions(cls) -> dict[str, str]:
        return {name: desc for name, (_, desc) in cls._registry.items()}
