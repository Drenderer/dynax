"""
This file implements some reoccuring typing used throughout the codebase.
"""

from typing import TypeVar, Protocol, Any

# Define a Protocol for callability
class CallableModule(Protocol):
    def __call__(self, *args, **kwargs) -> Any:
        ...

# The dollowing is a generic type annotation for a callable class that can be a stand-in for any equinox module.
Module = TypeVar('Module', bound=CallableModule)  # Self defined type for any equinox module subclass
