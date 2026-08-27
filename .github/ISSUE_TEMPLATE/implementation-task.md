---
name: Implementation Task
about: Standard ticket for assigning a module implementation to a project developer
title: "[TASK] "
labels: "implementation"
assignees: ''
---

## Project

<!-- Specify which project within the `magazine-03` repository this issue belongs to. -->

**Project:** 

## Context

<!-- Where this component fits in the overall architecture and why it is needed.
     Reference the relevant section(s) of the project's specification.
     Explain which other components depend on this one and how they interact. -->

## Interface to implement

<!-- The exact interface that this task must satisfy.
     Paste the full abstract method signatures here so the implementer has
     everything in one place without needing to navigate the codebase. -->

```python
# Example — replace with the real interface for this task

class IExample(ABC):
    @abstractmethod
    def example(self) -> None: ...