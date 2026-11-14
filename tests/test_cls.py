from functools import wraps
from typing import Any, cast
from unittest.mock import Mock

import pytest

from pyee import EventEmitter
from pyee.cls import evented, on


@evented
class EventedFixture:
    def __init__(self):
        self.call_me = Mock()

    @on("event")
    def event_handler(self, *args, **kwargs):
        self.call_me(self, *args, **kwargs)


_custom_event_emitter = EventEmitter()


@evented
class CustomEmitterFixture:
    def __init__(self):
        self.call_me = Mock()
        self.event_emitter = _custom_event_emitter

    @on("event")
    def event_handler(self, *args, **kwargs):
        self.call_me(self, *args, **kwargs)


class InheritedFixture(EventedFixture):
    pass


@pytest.mark.parametrize(
    "cls", [EventedFixture, CustomEmitterFixture, InheritedFixture]
)
def test_evented_decorator(cls):
    inst = cls()

    inst.event_emitter.emit("event", "emitter is emitted!")

    inst.call_me.assert_called_once_with(inst, "emitter is emitted!")

    _custom_event_emitter.remove_all_listeners()


@pytest.mark.parametrize("position", ["before", "after", "both"])
def test_nested_evented_classes_keep_their_own_handlers(position):
    @evented
    class Outer:
        event_emitter: EventEmitter

        def __init__(self):
            self.calls = []

        if position != "after":

            @on("outer_before")
            def before(self):
                self.calls.append("outer_before")

        @evented
        class Inner:
            event_emitter: EventEmitter

            def __init__(self):
                self.calls = []

            @on("inner")
            def handler(self):
                self.calls.append("inner")

        if position != "before":

            @on("outer_after")
            def after(self):
                self.calls.append("outer_after")

    outer, inner = Outer(), Outer.Inner()
    outer_events = {
        event
        for event, enabled in (
            ("outer_before", position != "after"),
            ("outer_after", position != "before"),
        )
        if enabled
    }
    assert outer.event_emitter.event_names() == outer_events
    assert inner.event_emitter.event_names() == {"inner"}
    for event in sorted(outer_events):
        assert outer.event_emitter.emit(event)
        assert not inner.event_emitter.emit(event)
    assert inner.event_emitter.emit("inner")
    assert not outer.event_emitter.emit("inner")
    assert outer.calls == sorted(outer_events)
    assert inner.calls == ["inner"]


def test_evented_preserves_handlers_for_later_class_decoration():
    class First:
        event_emitter: EventEmitter

        def __init__(self):
            self.calls = []

        @on("first")
        def handler(self):
            self.calls.append("first")

    @evented
    class Second:
        event_emitter: EventEmitter

        def __init__(self):
            self.calls = []

        @on("second")
        def handler(self):
            self.calls.append("second")

    evented(First)
    for cls, own, other in [(First, "first", "second"), (Second, "second", "first")]:
        instance: Any = cls()
        assert instance.event_emitter.emit(own)
        assert not instance.event_emitter.emit(other)
        assert instance.calls == [own]


def test_evented_alias_and_stacked_events_preserve_registration_order():
    @evented
    class Evented:
        event_emitter: EventEmitter

        def __init__(self):
            self.calls = []

        @on("one")
        @on("two")
        def handler(self, value):
            self.calls.append(value)

        alias = handler
        del handler

    instance = Evented()
    assert instance.event_emitter.emit("one", 1)
    assert instance.event_emitter.emit("two", 2)
    assert instance.calls == [1, 2]


def test_evented_inherited_handlers_and_custom_emitter():
    @evented
    class Parent:
        def __init__(self):
            self.event_emitter = EventEmitter()
            self.calls = []

        @on("event")
        def parent(self):
            self.calls.append("parent")

    @evented
    class Child(Parent):
        @on("event")
        def child(self):
            self.calls.append("child")

    child = Child()
    assert child.event_emitter.emit("event")
    assert child.calls == ["parent", "child"]


@pytest.mark.parametrize("descriptor", [staticmethod, classmethod])
def test_evented_keeps_descriptor_wrapped_handler_behavior(descriptor):
    @evented
    class Evented:
        event_emitter: EventEmitter

        def __init__(self):
            self.calls = []

        @descriptor
        @on("event")
        def handler(owner):
            owner.calls.append("event")

    instance = Evented()
    assert instance.event_emitter.emit("event")
    assert instance.calls == ["event"]


def test_evented_does_not_invoke_unrelated_descriptors():
    class Descriptor:
        @property
        def __wrapped__(self):
            raise AssertionError(
                "class collection must not resolve wrapper descriptors"
            )

        def __get__(self, instance, owner):
            raise AssertionError("class collection must not bind descriptors")

    @evented
    class Evented:
        event_emitter: EventEmitter

        unrelated = Descriptor()

        @on("event")
        def handler(self):
            self.called = True

    instance = Evented()
    assert instance.event_emitter.emit("event")
    assert instance.called


@pytest.mark.parametrize("on_outermost", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
def test_evented_preserves_wrapped_handler_binding(on_outermost, depth):
    calls: list[Any] = []

    def decorate(method):
        for _ in range(depth):

            def wrap(function):
                @wraps(function)
                def wrapped(*args, **kwargs):
                    calls.append("wrapper")
                    return function(*args, **kwargs)

                return wrapped

            method = wrap(method)
        return method

    def register(method):
        if on_outermost:
            return on("event")(decorate(method))
        return decorate(on("event")(method))

    @evented
    class Outer:
        event_emitter: EventEmitter

        @register
        def handler(self):
            calls.append(self)

        @evented
        class Inner:
            event_emitter: EventEmitter

            @register
            def handler(self):
                calls.append(self)

    outer, inner = Outer(), Outer.Inner()
    for instance in (outer, inner):
        assert instance.event_emitter.emit("event")
        expected = ["wrapper"] * depth if on_outermost else []
        assert calls == [*expected, instance]
        calls.clear()


def test_evented_cyclic_wrapper_metadata_terminates():
    @evented
    class Evented:
        event_emitter: EventEmitter

        @on("event")
        def handler(self):
            self.called = True

        cast(Any, handler).__wrapped__ = handler

    instance = Evented()
    assert instance.event_emitter.emit("event")
    assert instance.called
